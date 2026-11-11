'''LEAF Phase 1 part A: scheduler timestep record tests (unittest, stdlib only).

Run from the candidate root:  python -m unittest tests.test_forcing -v
'''
from __future__ import annotations

import datetime as dt
import unittest
from fractions import Fraction

from leaf import forcing

E = forcing.ForcingError
UTC = dt.timezone.utc
MST = dt.timezone(dt.timedelta(hours=-7))


class _Shifty(dt.tzinfo):
    '''A non-fixed (DST-capable style) tzinfo: must be refused.'''

    def utcoffset(self, d):
        return dt.timedelta(hours=1)

    def dst(self, d):
        return dt.timedelta(0)

    def tzname(self, d):
        return 'shifty'


def rec(start, end, t=20.0, **kw):
    return forcing.TimestepRecord(start, end, t, **kw)


class TestRecord(unittest.TestCase):
    def test_validation(self):
        a = dt.datetime(2024, 3, 10, 6, 0, tzinfo=MST)
        b = a + dt.timedelta(hours=18)
        r = rec(a, b, 12.5, pet_mm_per_day=3.0, label='day')
        self.assertEqual(r.temperature_c, 12.5)
        self.assertEqual(r.pet_mm_per_day, 3.0)
        self.assertEqual(r.process_kind, 'interevent')
        self.assertTrue(r.active)
        self.assertEqual(r.utc_offset, dt.timedelta(hours=-7))
        bad = [
            dict(start=a.replace(tzinfo=None)),                           # naive
            dict(end=b.replace(tzinfo=None)),
            dict(start=a.replace(tzinfo=_Shifty())),                      # non-fixed zone
            dict(start=a.replace(tzinfo=UTC)),                            # offset mismatch inside one step
            dict(end=a),                                                  # zero dt
            dict(end=a - dt.timedelta(seconds=1)),                        # negative dt
            dict(start=a.date()),                                         # date without clock
            dict(temperature_c=float('nan')), dict(temperature_c=float('inf')), dict(temperature_c=-300.0),
            dict(temperature_c=True), dict(temperature_c='12'), dict(temperature_c=None),
            dict(pet_mm_per_day=-1.0), dict(pet_mm_per_day=float('nan')), dict(pet_mm_per_day=True),
            dict(process_kind='wind'), dict(process_kind='rain'), dict(process_kind='storm_recession'),
            dict(process_kind='water_event'), dict(process_kind='growth'), dict(process_kind=3),
            dict(gate='maybe'), dict(gate=True), dict(label=7),
        ]
        for case in bad:
            kw = dict(start=a, end=b, temperature_c=12.5, pet_mm_per_day=None, process_kind='interevent', gate='open', label=None)
            kw.update(case)
            with self.assertRaises(E, msg=repr(case)):
                forcing.TimestepRecord(**kw)
        with self.assertRaises(E):
            forcing.TimestepRecord(a, b, temperature_c=12.5, pet_mm_per_day=3.0, process_kind='interevent', gate='open').with_gate('x')
        c = r.with_gate('closed')
        self.assertFalse(c.active)
        self.assertEqual(c.temperature_c, 12.5)
        with self.assertRaises(Exception):
            r.temperature_c = 1.0                                           # frozen
        d = r.to_dict()
        self.assertEqual(d['utc_offset_seconds'], -7 * 3600)
        self.assertEqual(d['dt_seconds_exact'], '64800')

    def test_exact_fractions_and_partial_days(self):
        a = dt.datetime(2024, 5, 1, 10, 15, 0, 250_000, tzinfo=UTC)
        r = rec(a, a + dt.timedelta(hours=1, minutes=30))
        self.assertEqual(r.dt_seconds_exact, Fraction(5400))
        self.assertEqual(r.dt_days_exact, Fraction(1, 16))
        self.assertEqual(r.dt_days, 1.0 / 16.0)
        r2 = rec(a, a + dt.timedelta(microseconds=1))
        self.assertEqual(r2.dt_seconds_exact, Fraction(1, 10 ** 6))
        self.assertGreater(r2.dt_seconds, 0.0)
        self.assertFalse(r.crosses_local_midnight)
        self.assertFalse(r.is_whole_local_days)
        r3 = rec(dt.datetime(2024, 5, 1, tzinfo=MST), dt.datetime(2024, 5, 3, tzinfo=MST))
        self.assertTrue(r3.is_whole_local_days)
        self.assertEqual(r3.dt_days_exact, Fraction(2))
        self.assertEqual(r3.end_date, dt.date(2024, 5, 2))                  # an end at midnight belongs to the day before

    def test_leap_day_year_boundary_and_midnight_split(self):
        leap = rec(dt.datetime(2024, 2, 28, 12, tzinfo=MST), dt.datetime(2024, 3, 1, 12, tzinfo=MST))
        self.assertEqual(leap.dt_days_exact, Fraction(2))                   # 2024-02-29 exists
        common = rec(dt.datetime(2023, 2, 28, 12, tzinfo=MST), dt.datetime(2023, 3, 1, 12, tzinfo=MST))
        self.assertEqual(common.dt_days_exact, Fraction(1))
        pieces = leap.split_at_local_midnight()
        self.assertEqual([p.start.date() for p in pieces], [dt.date(2024, 2, 28), dt.date(2024, 2, 29), dt.date(2024, 3, 1)])
        self.assertEqual(sum(p.dt_days_exact for p in pieces), Fraction(2))
        self.assertEqual(forcing.check_contiguous(pieces)['total_days_exact'], '2')
        ny = rec(dt.datetime(2023, 12, 31, 18, tzinfo=MST), dt.datetime(2024, 1, 1, 6, tzinfo=MST), -3.0, pet_mm_per_day=0.5)
        parts = ny.split_at_local_midnight()
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0].end, dt.datetime(2024, 1, 1, 0, tzinfo=MST))
        self.assertEqual(parts[1].start.date().year, 2024)
        self.assertEqual([p.dt_days_exact for p in parts], [Fraction(1, 4), Fraction(1, 4)])
        self.assertTrue(all(p.temperature_c == -3.0 and p.pet_mm_per_day == 0.5 for p in parts))
        # the local calendar is the fixed offset's clock: the same instants expressed in UTC (01:00 -> 13:00 on
        # 2024-01-01) cross no UTC midnight, so they are not split, while the MST record is
        in_utc = rec(ny.start.astimezone(UTC), ny.end.astimezone(UTC), -3.0)
        self.assertTrue(ny.crosses_local_midnight)
        self.assertFalse(in_utc.crosses_local_midnight)
        self.assertEqual(len(in_utc.split_at_local_midnight()), 1)
        self.assertEqual(in_utc.dt_seconds_exact, ny.dt_seconds_exact)
        # a step ending exactly at midnight is not split
        self.assertEqual(len(rec(dt.datetime(2024, 1, 1, 6, tzinfo=MST), dt.datetime(2024, 1, 2, 0, tzinfo=MST)).split_at_local_midnight()), 1)

    def test_growth_boundary_split_and_contiguity(self):
        a = dt.datetime(2024, 6, 1, tzinfo=UTC)
        r = rec(a, a + dt.timedelta(days=3), 25.0)
        inside = a + dt.timedelta(days=1, hours=6)
        outside = a + dt.timedelta(days=9)
        parts = r.split_at([outside, inside, a, r.end])                    # start / end / outside boundaries ignored
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0].end, inside)
        self.assertEqual(parts[1].start, inside)
        self.assertEqual(sum(p.dt_seconds_exact for p in parts), r.dt_seconds_exact)
        with self.assertRaises(E):
            r.split_at([inside.astimezone(MST)])                            # boundary offset must match
        with self.assertRaises(E):
            r.split_at([inside.replace(tzinfo=None)])
        seq = [rec(a, a + dt.timedelta(hours=6)), rec(a + dt.timedelta(hours=6), a + dt.timedelta(days=1), gate='closed')]
        s = forcing.check_contiguous(seq)
        self.assertEqual((s['n_records'], s['n_gate_closed']), (2, 1))
        self.assertEqual(s['total_days_exact'], '1')
        with self.assertRaises(E):
            forcing.check_contiguous([seq[1], seq[0]])                      # reordered
        with self.assertRaises(E):
            forcing.check_contiguous([seq[0], rec(a + dt.timedelta(hours=7), a + dt.timedelta(days=1))])   # gap: no catch-up
        with self.assertRaises(E):
            forcing.check_contiguous([seq[0], rec(a + dt.timedelta(hours=5), a + dt.timedelta(days=1))])   # overlap
        with self.assertRaises(E):
            forcing.check_contiguous([seq[0], rec(seq[0].end.astimezone(MST), a + dt.timedelta(days=1))])   # offset change
        with self.assertRaises(E):
            forcing.check_contiguous([])
        with self.assertRaises(E):
            forcing.check_contiguous([seq[0], 'x'])

    def test_weather_rows_conversion(self):
        rows = [{'date': '2024-02-28', 'tmean_c': 4.0, 'pet': 1.5},
                {'date': '2024-02-29', 'tmean_c': 5.5, 'pet': 1.0},
                {'date': '2024-03-01', 'tmean_c': 6.0, 'pet': 2.0}]
        recs = forcing.records_from_weather_rows(rows, dt.timedelta(hours=-7), pet_key='pet')
        self.assertEqual(len(recs), 3)
        self.assertEqual(recs[0].start, dt.datetime(2024, 2, 28, tzinfo=MST))
        self.assertEqual(recs[1].start.date(), dt.date(2024, 2, 29))
        self.assertEqual(recs[2].end, dt.datetime(2024, 3, 2, tzinfo=MST))
        self.assertEqual([r.temperature_c for r in recs], [4.0, 5.5, 6.0])
        self.assertEqual([r.pet_mm_per_day for r in recs], [1.5, 1.0, 2.0])
        self.assertTrue(all(r.is_whole_local_days for r in recs))
        self.assertEqual(forcing.check_contiguous(recs)['total_days_exact'], '3')
        no_pet = forcing.records_from_weather_rows(rows[:1], MST)
        self.assertIsNone(no_pet[0].pet_mm_per_day)
        bad_rows = [
            [{'date': '2023-02-29', 'tmean_c': 1.0}],                                      # not a leap year
            [{'date': '2024-2-29', 'tmean_c': 1.0}],                                       # not zero padded
            [{'date': '2024/02/29', 'tmean_c': 1.0}],
            [{'date': 20240229, 'tmean_c': 1.0}],
            [{'date': '2024-02-28', 'pet': 1.0}],                                          # temperature required even with PET
            [{'date': '2024-02-28', 'tmean_c': 'warm'}],
            [{'date': '2024-02-28', 'tmean_c': 1.0}, {'date': '2024-03-01', 'tmean_c': 1.0}],   # gap (leap day missing)
            [{'date': '2024-02-28', 'tmean_c': 1.0}, {'date': '2024-02-28', 'tmean_c': 1.0}],   # duplicate
            [{'date': '2024-02-29', 'tmean_c': 1.0}, {'date': '2024-02-28', 'tmean_c': 1.0}],   # reordered
            [],
            [('2024-02-28', 1.0)],
        ]
        for rows_bad in bad_rows:
            with self.assertRaises(E, msg=repr(rows_bad)):
                forcing.records_from_weather_rows(rows_bad, MST)
        with self.assertRaises(E):
            forcing.records_from_weather_rows(rows, MST, pet_key='missing')
        with self.assertRaises(E):
            forcing.records_from_weather_rows(rows, -7)                                       # offset must be explicit
        with self.assertRaises(E):
            forcing.records_from_weather_rows(rows, MST, gate='on')


if __name__ == '__main__':
    unittest.main()
