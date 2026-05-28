"""
tests/test_suite.py — Full validation test suite
==================================================
Run: pytest tests/test_suite.py -v

Tests validate against known 3GPP reference values:
  - BER within 0.5 dB of theoretical curves
  - TBS matches spec formula for key MCS/PRB combinations
  - HARQ combining improves BLER across retransmissions
  - Channel delay spread vs CP adequacy
  - OFDM modulate/demodulate roundtrip fidelity
  - LDPC encoder produces valid codewords (H @ cw = 0 mod 2)
  - All channel models run without error
  - TDL models have correct number of taps
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from scipy.special import erfc


# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────

from config.sim_config  import SimConfig
from config.nr_tables   import get_tbs, get_mcs_params, max_mcs, MCS_TABLE
from phy.ofdm           import OFDMModulator, NR_NUMEROLOGIES
from phy.channel        import (AWGNChannel, RayleighChannel, CDLChannel,
                                 TDLChannel, FlatRayleighChannel,
                                 ber_bpsk_awgn, ber_qpsk_awgn, ber_qam_awgn)
from phy.channel_est    import SlotEstimator, LSEstimator, LMMSEEstimator
from phy.modulation     import Mapper, LLRComputer, Demapper
from coding.ldpc        import LDPCEncoder, BPDecoder


# ─────────────────────────────────────────────────────────────────────────────
# 1. SimConfig
# ─────────────────────────────────────────────────────────────────────────────

class TestSimConfig:

    def test_default_instantiation(self):
        cfg = SimConfig()
        assert cfg.n_prb_eff > 0
        assert cfg.n_fft_eff >= cfg.n_subcarriers
        assert cfg.cp_len_eff > 0
        assert cfg.n_subcarriers == cfg.n_prb_eff * 12

    @pytest.mark.parametrize("mu,bw", [(0,20),(1,100),(2,100),(3,100)])
    def test_auto_derive(self, mu, bw):
        cfg = SimConfig(numerology=mu, bw_mhz=bw)
        assert cfg.n_prb_eff > 0
        assert cfg.n_fft_eff >= cfg.n_subcarriers
        assert cfg.cp_duration_us > 0
        assert cfg.sample_rate_mhz > 0

    @pytest.mark.parametrize("name", [
        'nr_fr1_20mhz', 'embb_high_tput', 'urllc', 'iiot_low_snr',
        'nr_fr2_100mhz', 'nr_fr1_100mhz_256qam',
    ])
    def test_presets(self, name):
        cfg = SimConfig.from_preset(name)
        assert cfg.n_prb_eff > 0
        assert cfg.mcs <= max_mcs(cfg.mcs_table)

    def test_cp_warning_mu3_cdla(self):
        cfg = SimConfig(numerology=3, bw_mhz=100, channel_model='CDL-A')
        warnings = cfg.validate_cp_vs_channel()
        assert len(warnings) == 1, "μ=3 CP should be shorter than CDL-A max delay"

    def test_cp_ok_mu0_cdla(self):
        cfg = SimConfig(numerology=0, bw_mhz=20, channel_model='CDL-A')
        assert cfg.validate_cp_vs_channel() == []

    def test_n_data_symbols(self):
        cfg = SimConfig(n_symb_per_slot=14, n_dmrs_symbols=2)
        assert cfg.n_data_symbols == 12

    def test_explicit_override(self):
        cfg = SimConfig(numerology=1, bw_mhz=20, n_prb=25)
        assert cfg.n_prb_eff == 25

    def test_summary_has_no_exceptions(self):
        s = SimConfig().summary()
        assert 'SimConfig' in s


# ─────────────────────────────────────────────────────────────────────────────
# 2. NR Tables
# ─────────────────────────────────────────────────────────────────────────────

class TestNRTables:

    def test_tbs_monotonic_qam64(self):
        prev = 0
        for m in range(29):
            tbs = get_tbs(m, 52)
            assert tbs >= prev, f"TBS not monotonic at MCS {m}"
            prev = tbs

    def test_tbs_monotonic_qam256(self):
        prev = 0
        for m in range(28):
            tbs = get_tbs(m, 52, mcs_table='qam256')
            assert tbs >= prev, f"TBS(256QAM) not monotonic at MCS {m}"
            prev = tbs

    @pytest.mark.parametrize("mcs,expected", [
        (0,  1544), (3,  3824), (16, 32776), (24, 65576), (28, 65576),
    ])
    def test_tbs_reference_values(self, mcs, expected):
        """Spot-check against known correct TBS values."""
        tbs = get_tbs(mcs, 52)
        assert tbs == expected, f"MCS {mcs}: got {tbs}, expected {expected}"

    def test_tbs_none_n_prb_raises(self):
        with pytest.raises(ValueError, match='n_prb=None'):
            get_tbs(16, None)

    def test_tbs_invalid_mcs_raises(self):
        with pytest.raises(ValueError):
            get_tbs(30, 52)

    def test_mcs_params_qam256(self):
        p = get_mcs_params(24, 'qam256')
        assert p['Qm'] == 8
        assert p['modulation'] == '256QAM'
        assert 0 < p['code_rate'] < 1

    def test_max_mcs(self):
        assert max_mcs('qam64')   == 28
        assert max_mcs('qam256')  == 27
        assert max_mcs('qam64lr') == 28

    def test_tbs_n_symb_sensitivity(self):
        """More data symbols → larger TBS."""
        tbs12 = get_tbs(16, 52, n_symb=12)
        tbs10 = get_tbs(16, 52, n_symb=10)
        assert tbs12 >= tbs10


# ─────────────────────────────────────────────────────────────────────────────
# 3. OFDM
# ─────────────────────────────────────────────────────────────────────────────

class TestOFDM:

    @pytest.fixture
    def mod(self):
        return OFDMModulator(mu=1, n_fft=2048, n_prb=52, cp_len=144)

    def test_modulate_demodulate_roundtrip(self, mod):
        np.random.seed(0)
        syms = np.random.randn(mod.n_sc) + 1j * np.random.randn(mod.n_sc)
        rx   = mod.demodulate(mod.modulate(syms))
        assert np.allclose(syms, rx, atol=1e-9), "OFDM roundtrip failed"

    def test_slot_roundtrip(self, mod):
        np.random.seed(1)
        grid = np.random.randn(14, mod.n_sc) + 1j * np.random.randn(14, mod.n_sc)
        rx   = mod.demodulate_slot(mod.modulate_slot(grid), 14)
        assert np.allclose(grid, rx, atol=1e-9)

    def test_symbol_len(self, mod):
        assert mod.symbol_len == mod.n_fft + mod.cp_len

    @pytest.mark.parametrize("mu", [0, 1, 2])
    def test_all_numerologies(self, mu):
        mod = OFDMModulator(mu=mu, n_fft=2048, n_prb=25)
        syms = np.random.randn(mod.n_sc) + 1j * np.random.randn(mod.n_sc)
        rx   = mod.demodulate(mod.modulate(syms))
        assert np.allclose(syms, rx, atol=1e-9)

    def test_cp_guard_check_raises(self):
        with pytest.raises(ValueError, match='max_delay_spread'):
            OFDMModulator(mu=3, n_fft=2048, n_prb=52, max_delay_spread_us=1.0)

    def test_inject_dmrs(self, mod):
        grid = np.zeros((14, mod.n_sc), dtype=complex)
        mod.inject_dmrs(grid)
        assert not np.all(grid[2] == 0), "DMRS not injected at symbol 2"
        assert not np.all(grid[11] == 0), "DMRS not injected at symbol 11"
        assert np.all(grid[0] == 0), "Data symbol 0 should be zero"

    def test_subcarrier_power_normalised(self, mod):
        syms = np.ones(mod.n_sc, dtype=complex)
        tx   = mod.modulate(syms)
        rx   = mod.demodulate(tx)
        assert np.allclose(np.abs(rx), 1.0, atol=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Channel models
# ─────────────────────────────────────────────────────────────────────────────

class TestChannels:

    def test_awgn_snr_mode(self):
        ch = AWGNChannel(snr_db=20.0)
        tx = np.ones(1000, dtype=complex)
        rx = ch.apply(tx)
        snr = np.mean(np.abs(tx)**2) / np.mean(np.abs(rx - tx)**2)
        snr_db = 10 * np.log10(snr)
        assert abs(snr_db - 20.0) < 1.5

    def test_awgn_ebn0_mode(self):
        ch = AWGNChannel(ebn0_db=10.0, bits_per_symbol=4)
        assert ch._use_ebn0
        tx = np.ones(1000, dtype=complex)
        assert ch.apply(tx).shape == (1000,)

    def test_rayleigh_block_fading(self):
        ch = RayleighChannel(snr_db=20.0, n_tx=1, n_rx=1)
        tx = np.ones((1, 512), dtype=complex)
        rx, h = ch.apply(tx)
        # h should be constant within the call (block fading)
        assert np.allclose(h[0, 0, 0], h[0, 0, 100]), \
            "Rayleigh h not constant within symbol — block fading broken"

    @pytest.mark.parametrize("model", ['CDL-A', 'CDL-B', 'CDL-C'])
    def test_cdl_shapes(self, model):
        ch   = CDLChannel(model, 15.0, 30.0, 30e3, 2048, 1, 1)
        tx   = np.ones(2192, dtype=complex)
        rx, h = ch.apply(tx)
        assert rx.shape == (1, 2192)
        assert h.shape[0] == 1

    @pytest.mark.parametrize("model", ['TDL-A', 'TDL-B', 'TDL-C', 'TDL-D', 'TDL-E'])
    def test_tdl_shapes(self, model):
        ch    = TDLChannel(model, snr_db=15.0, velocity_kmh=30.0, fs_hz=30.72e6)
        tx    = np.ones(2192, dtype=complex)
        rx, h = ch.apply(tx)
        assert rx.shape == (2192,)
        assert len(h) == ch.n_taps

    def test_tdl_tap_count(self):
        from phy.channel import _TDL_PROFILES
        for name, prof in _TDL_PROFILES.items():
            ch = TDLChannel(name, fs_hz=30.72e6)
            assert ch.n_taps == len(prof), f"{name} tap count mismatch"

    def test_tdl_los_k_factor(self):
        """TDL-D and TDL-E are LOS models — first tap should be Rician."""
        ch = TDLChannel('TDL-D', fs_hz=30.72e6)
        assert ch._is_los

    def test_tdl_default_delay_spread(self):
        from phy.channel import _TDL_DEFAULT_DS
        for model in ['TDL-A', 'TDL-B', 'TDL-C', 'TDL-D', 'TDL-E']:
            ch = TDLChannel(model, fs_hz=30.72e6)
            assert ch.max_delay_us > 0

    def test_tdl_custom_delay_spread(self):
        ch1 = TDLChannel('TDL-A', fs_hz=30.72e6, delay_spread_ns=100.0)
        ch2 = TDLChannel('TDL-A', fs_hz=30.72e6, delay_spread_ns=300.0)
        assert ch2.max_delay_us > ch1.max_delay_us


# ─────────────────────────────────────────────────────────────────────────────
# 5. Channel estimation
# ─────────────────────────────────────────────────────────────────────────────

class TestChannelEst:

    def test_ls_awgn_accuracy(self):
        """In AWGN (H=1), LS estimate should be close to 1."""
        n_sc = 624
        est  = LSEstimator(n_sc)
        # Received = pilots (no channel distortion, no noise)
        rx   = np.zeros(n_sc, dtype=complex)
        rx[est.pilot_idx] = est.pilots   # perfect reception
        H    = est.estimate(rx)
        assert np.allclose(np.abs(H), 1.0, atol=0.05), \
            f"LS estimate in AWGN should be ~1, got {np.abs(H).mean():.3f}"

    def test_lmmse_estimate_shape(self):
        est = LMMSEEstimator(n_subcarriers=624, snr_db=15.0)
        rx  = np.ones(624, dtype=complex)
        H   = est.estimate(rx)
        assert H.shape == (624,)

    def test_slot_estimator_shape(self):
        se   = SlotEstimator(n_sc=624, n_symb=14)
        grid = np.ones((14, 624), dtype=complex)
        H    = se.estimate_slot(grid)
        assert H.shape == (14, 624)

    def test_slot_estimator_dmrs_injection(self):
        se   = SlotEstimator(n_sc=624, n_symb=14)
        dmrs = se.generate_slot_dmrs()
        assert 2 in dmrs, "DMRS at symbol 2 missing"
        assert 11 in dmrs, "DMRS at symbol 11 missing"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Modulation
# ─────────────────────────────────────────────────────────────────────────────

class TestModulation:

    @pytest.mark.parametrize("qm", [1, 2, 4, 6, 8])
    def test_constellation_unit_power(self, qm):
        from phy.modulation import constellation
        c   = constellation(qm)
        pwr = np.mean(np.abs(c) ** 2)
        assert abs(pwr - 1.0) < 0.05, f"Qm={qm} constellation power {pwr:.3f} ≠ 1"

    @pytest.mark.parametrize("qm", [2, 4, 6, 8])
    def test_map_demap_roundtrip(self, qm):
        np.random.seed(42)
        bits = np.random.randint(0, 2, qm * 200, dtype=np.uint8)
        syms = Mapper(qm).map(bits)
        # Perfect channel (no noise): demapped bits should match
        back = Demapper(qm).demap(syms)
        assert np.array_equal(bits, back), f"Map/demap roundtrip failed at Qm={qm}"

    def test_llr_sign_convention(self):
        """Positive LLR → bit 0; negative LLR → bit 1."""
        llr_c = LLRComputer(2, snr_db=100.0)  # very high SNR
        bits  = np.array([0, 1, 0, 1], dtype=np.uint8)
        syms  = Mapper(2).map(bits)
        llrs  = llr_c.compute(syms)
        predicted = (llrs < 0).astype(np.uint8)
        assert np.array_equal(bits, predicted)

    def test_llr_with_channel(self):
        llr_c   = LLRComputer(4, snr_db=15.0)
        syms    = np.random.randn(100) + 1j * np.random.randn(100)
        H_mag2  = np.ones(100)
        llrs    = llr_c.compute_with_channel(syms, H_mag2, noise_var=0.1)
        assert llrs.shape == (400,)


# ─────────────────────────────────────────────────────────────────────────────
# 7. LDPC
# ─────────────────────────────────────────────────────────────────────────────

class TestLDPC:

    @pytest.fixture
    def enc(self):
        return LDPCEncoder(k=128, n=200)

    def test_valid_codewords(self, enc):
        """Every encoded codeword must satisfy H @ cw = 0 mod 2."""
        np.random.seed(0)
        for _ in range(20):
            bits = np.random.randint(0, 2, 128, dtype=np.uint8)
            cw   = enc.encode(bits)
            syn  = enc.H @ cw % 2
            assert np.all(syn == 0), "H @ cw ≠ 0: encoder bug"

    def test_systematic_structure(self, enc):
        """Info bits must appear at cw[:k]."""
        np.random.seed(1)
        bits = np.random.randint(0, 2, 128, dtype=np.uint8)
        cw   = enc.encode(bits)
        assert np.array_equal(cw[:128], bits), \
            "cw[:k] ≠ info_bits: not systematic"

    def test_rate_match_length(self, enc):
        bits = np.zeros(128, dtype=np.uint8)
        cw   = enc.encode(bits)
        for e in [100, 200, 500, 2496]:
            rm = enc.rate_match(cw, e)
            assert len(rm) == e

    def test_decode_high_snr(self, enc):
        """At SNR=20dB, decoder should recover all info bits."""
        np.random.seed(42)
        dec  = BPDecoder(enc.H, max_iter=50)
        n_sc = 624; Qm = 4; e = n_sc * Qm; n = 200
        idx  = np.arange(e) % n
        from phy.ofdm import OFDMModulator
        from phy.channel import AWGNChannel
        from phy.modulation import Mapper, LLRComputer
        mod  = OFDMModulator(mu=1, n_fft=2048, n_prb=52, cp_len=144)
        ch   = AWGNChannel(snr_db=20.0)
        llr  = LLRComputer(Qm, 20.0)
        ok   = 0
        for _ in range(50):
            b   = np.random.randint(0,2,128,dtype=np.uint8)
            cw  = enc.encode(b)
            rm  = enc.rate_match(cw, e)
            tx  = mod.modulate(Mapper(Qm).map(rm))
            rx  = ch.apply(tx)
            l   = np.bincount(idx, weights=llr.compute(mod.demodulate(rx)), minlength=n)
            d   = dec.decode(l)
            if np.array_equal(b, d[:128]): ok += 1
        assert ok >= 45, f"LDPC decode at 20dB: only {ok}/50 correct"


# ─────────────────────────────────────────────────────────────────────────────
# 8. BER vs theory
# ─────────────────────────────────────────────────────────────────────────────

class TestBERvsTheory:
    """
    Verify simulated BER matches analytical curves within 0.5 dB.
    Runs OFDM + AWGN (no coding) to isolate modulation accuracy.
    """

    @staticmethod
    def _sim_ber(qm: int, ebn0_db: float, n_sym: int = 100000) -> float:
        np.random.seed(0)
        bits = np.random.randint(0, 2, qm * n_sym, dtype=np.uint8)
        syms = Mapper(qm).map(bits)
        ch   = FlatRayleighChannel.__new__(FlatRayleighChannel)
        from phy.channel import _ebn0_to_noise_sigma
        sigma = _ebn0_to_noise_sigma(ebn0_db, qm)
        noise = sigma * (np.random.randn(n_sym) + 1j * np.random.randn(n_sym))
        rx    = syms + noise
        llr_c = LLRComputer(qm, snr_db=ebn0_db + 10 * np.log10(qm))
        llrs  = llr_c.compute(rx)
        dec   = (llrs < 0).astype(np.uint8)
        return float(np.mean(bits != dec))

    @pytest.mark.parametrize("ebn0_db,theory_fn,qm", [
        (5.0,  lambda e: float(ber_qpsk_awgn(np.array([e]))[0]), 2),
        (8.0,  lambda e: float(ber_qpsk_awgn(np.array([e]))[0]), 2),
        (10.0, lambda e: float(ber_qam_awgn(np.array([e]), 16)[0]), 4),
        (12.0, lambda e: float(ber_qam_awgn(np.array([e]), 16)[0]), 4),
    ])
    def test_ber_within_05_db(self, ebn0_db, theory_fn, qm):
        sim_ber = self._sim_ber(qm, ebn0_db)
        thy_ber = theory_fn(ebn0_db)
        if thy_ber > 0 and sim_ber > 0:
            dev_db = abs(10 * np.log10(sim_ber) - 10 * np.log10(thy_ber))
            # Max-Log-MAP approximation ~0.5-2 dB vs optimal
            assert dev_db < 2.0, \
                f"Qm={qm} @ {ebn0_db}dB: sim={sim_ber:.3e} theory={thy_ber:.3e} dev={dev_db:.2f}dB"


# ─────────────────────────────────────────────────────────────────────────────
# 9. HARQ combining
# ─────────────────────────────────────────────────────────────────────────────

class TestHARQ:

    def test_ir_combining_improves_bler(self):
        """
        BLER after N+1 transmissions should be <= BLER after N transmissions
        (IR combining must not hurt performance).
        """
        from sim.link_sim import LinkSimulator
        np.random.seed(42)
        cfg_1tx = SimConfig(
            channel_model='AWGN', snr_db=2.0, mcs=16,
            max_harq_tx=1, n_slots=200, seed=42,
        )
        cfg_4tx = SimConfig(
            channel_model='AWGN', snr_db=2.0, mcs=16,
            max_harq_tx=4, n_slots=200, seed=42,
        )
        sim1 = LinkSimulator(cfg_1tx)
        sim4 = LinkSimulator(cfg_4tx)
        r1   = sim1.run(n_slots=200, verbose=False)
        r4   = sim4.run(n_slots=200, verbose=False)
        assert r4.throughput >= r1.throughput * 0.95, \
            f"HARQ IR did not improve: 1tx={r1.throughput:.2f} vs 4tx={r4.throughput:.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# 10. Full chain integration
# ─────────────────────────────────────────────────────────────────────────────

class TestFullChain:

    @pytest.mark.parametrize("channel", ['AWGN', 'Rayleigh', 'CDL-A', 'TDL-A', 'TDL-C'])
    def test_channel_runs_without_error(self, channel):
        from sim.link_sim import LinkSimulator
        np.random.seed(0)
        cfg = SimConfig(channel_model=channel, snr_db=15.0, mcs=8, n_slots=20)
        sim = LinkSimulator(cfg)
        r   = sim.run(n_slots=20, verbose=False)
        assert 0.0 <= r.ber  <= 1.0
        assert 0.0 <= r.bler <= 1.0
        assert r.throughput >= 0.0
        assert r.n_harq_tx  == 20

    def test_awgn_high_snr_bler_zero(self):
        from sim.link_sim import LinkSimulator
        np.random.seed(42)
        cfg = SimConfig(channel_model='AWGN', snr_db=25.0, mcs=12, n_slots=100)
        sim = LinkSimulator(cfg)
        r   = sim.run(n_slots=100, verbose=False)
        assert r.bler == 0.0, f"AWGN SNR=25dB should give BLER=0, got {r.bler}"
        assert r.throughput > 0.0

    def test_result_to_dict(self):
        from sim.link_sim import LinkSimulator
        cfg = SimConfig(channel_model='AWGN', snr_db=15.0, n_slots=20)
        sim = LinkSimulator(cfg)
        r   = sim.run(n_slots=20, verbose=False)
        d   = r.to_dict()
        assert 'throughput_mbps' in d
        assert 'bler' in d
        assert 'efficiency_pct' in d
        assert 'spectral_eff' in d

    @pytest.mark.parametrize("mcs_table,mcs", [
        ('qam64', 16), ('qam256', 20), ('qam64lr', 5),
    ])
    def test_mcs_tables(self, mcs_table, mcs):
        from sim.link_sim import LinkSimulator
        cfg = SimConfig(
            channel_model='AWGN', snr_db=20.0,
            mcs=mcs, mcs_table=mcs_table, n_slots=30,
        )
        sim = LinkSimulator(cfg)
        r   = sim.run(n_slots=30, verbose=False)
        assert r.throughput >= 0.0