'''Tests for leaf.biocalendar (reused qualified C1 calendar functions) at the daily cadence used by the plant session. Pure; NOT RUN by the
author. Run from the candidate root: python -m unittest tests.test_biocalendar -v'''
import datetime as dt
import os
import sys
import unittest
from fractions import Fraction

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from leaf import biocalendar as bio, forcing  # noqa: E402

UTC = dt.timezone.utc
A = dt.datetime(2027, 6, 1, tzinfo=UTC)
DAY = dt.timedelta(days=1)


def rec(i, n=1):
    return forcing.TimestepRecord(A + i * DAY, A + (i + n) * DAY, 20.0)


class CadenceTests(unittest.TestCase):
    def test_seven_one_day_steps_fire_exactly_at_the_grid_point(self):
        s = bio.new_schedule(A, 7 * DAY)
        fired = []
        for i in range(14):
            st = bio.open_step(s, rec(i))
            fired.append(st.fire_growth)
            self.assertEqual(st.credit_days, Fraction(1))
            s = st.schedule
        self.assertEqual([i for i, f in enumerate(fired) if f], [6, 13])
        self.assertEqual(s.pending_days, 0)

    def test_growth_disabled_adds_no_credit_and_keeps_earned_credit(self):
        s = bio.new_schedule(A, 7 * DAY)
        s = bio.open_step(s, rec(0)).schedule
        s2 = bio.open_step(s, rec(1), growth_enabled=False).schedule
        self.assertEqual(s2.pending_days, s.pending_days)
        self.assertEqual(s2.cursor, A + 2 * DAY)
        with self.assertRaises(bio.BioCalendarStateError):
            bio.flush_step(s2, growth_enabled=False)                                 # never flush inside a disabled period
        self.assertTrue(bio.flush_required(s2, next_growth_enabled=False, next_is_transport=False))
        self.assertEqual(bio.flush_step(s2, growth_enabled=True).pending_days, 0)

    def test_transport_requires_flush_and_skips_boundaries_without_catch_up(self):
        s = bio.new_schedule(A, 3 * DAY)
        s = bio.open_step(s, rec(0)).schedule
        with self.assertRaises(bio.BioCalendarStateError):
            bio.transport_step(s, s.cursor, s.cursor + 5 * DAY)
        s = bio.flush_step(s, growth_enabled=True)
        ts = bio.transport_step(s, s.cursor, s.cursor + 5 * DAY)
        self.assertGreaterEqual(ts.skipped_boundaries, 1)
        self.assertEqual(ts.schedule.pending_days, 0)

    def test_year_end_and_leap_year_length(self):
        self.assertEqual(bio.year_days(dt.datetime(2028, 3, 1, tzinfo=UTC)), 366)
        self.assertEqual(bio.year_days(dt.datetime(2027, 3, 1, tzinfo=UTC)), 365)
        a = dt.datetime(2027, 12, 31, tzinfo=UTC)
        s = bio.new_schedule(a, 7 * DAY)
        st = bio.open_step(s, forcing.TimestepRecord(a, a + DAY, 5.0))
        self.assertTrue(st.year_end)
        self.assertTrue(st.fire_growth)

    def test_replay_gap_and_unsplit_refused(self):
        s = bio.new_schedule(A, 7 * DAY)
        s1 = bio.open_step(s, rec(0)).schedule
        with self.assertRaises(bio.BioCalendarStateError):
            bio.open_step(s1, rec(0))                                                # replay
        with self.assertRaises(bio.BioCalendarStateError):
            bio.open_step(s1, rec(3))                                                # gap: no catch-up
        with self.assertRaises(bio.BioCalendarBoundaryError):
            bio.open_step(s1, rec(1, 2))                                             # crosses a local midnight: split first

    def test_cadence_above_seven_days_is_refused(self):
        with self.assertRaises(bio.BioCalendarError):
            bio.new_schedule(A, 8 * DAY)

    def test_credit_increment_disabled_returns_same_arrays_and_enabled_adds(self):
        D, U = np.array([1.0, 2.0]), np.array([0.5, 2.0])
        Ai, Ti = np.array([0.001]), np.array([0.0005])
        out = bio.credit_increment(D, U, Ai, Ti, D, U, Ai, Ti, growth_enabled=False, max_bytes=1 << 20)
        self.assertTrue(all(o is x for o, x in zip(out, (D, U, Ai, Ti))))
        Dn, Un, An, Tn = bio.credit_increment(D, U, Ai, Ti, D, U, Ai, Ti, growth_enabled=True, max_bytes=1 << 20)
        np.testing.assert_array_equal(Dn, 2 * D)
        with self.assertRaises(bio.BioCalendarError):
            bio.credit_increment(D, U, Ai, Ti, D, U + 5.0, Ai, Ti, growth_enabled=True, max_bytes=1 << 20)   # U > D
        with self.assertRaises(bio.BioCalendarAdmissionError):
            bio.credit_increment(D, U, Ai, Ti, D, U, Ai, Ti, growth_enabled=True, max_bytes=1)


if __name__ == '__main__':
    unittest.main()
