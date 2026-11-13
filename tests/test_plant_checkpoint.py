'''Tests for leaf.plant_checkpoint (biological-only save / restore, version 2). Synthetic LEAF-side water doubles; NOT RUN by the author.
Run from the candidate root: python -m unittest tests.test_plant_checkpoint -v'''
import dataclasses
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leaf import plant_checkpoint as pc, plant_session as ps  # noqa: E402
import native_leaf_fixture as nf  # noqa: E402

LIM = pc.CheckpointLimits(1 << 26)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.cat = nf.make_catalog()
        self.cfg = nf.make_config(self.cat)
        self.a = nf.anchor()
        s = nf.start_session(nf.make_state(self.cat), self.cfg, self.a)
        for i in range(3):
            s = self.day(s, i)
        self.s = s
        self.dir = os.path.join(self.tmp, 'ck')
        self.ext = s.resource_cursor          # the exact external reference: the state after the last consumed T (a boundary closed after it
        #                                      would have another digest; any exact ResourceIdentity is paired the same way)

    def day(self, s, i, factor=1.0):
        kw = nf.next_read_args(s)
        plan = s.prepare_daily(nf.day_ctx(self.a, i), terrain_pet=nf.pet_field(s.pop.geometry.root_map.n_cells, 5.0))
        prop = s.propose_withdrawal(plan, nf.make_read(s, ordinal=i, run=nf.HEX_D, **kw))
        return s.publish(s.stage_day(prop, nf.make_accepted(prop, factor)))

    def save(self, s=None, **kw):
        return pc.save_biology(self.dir, s or self.s, external=kw.pop('external', self.ext), limits=kw.pop('limits', LIM), **kw)

    def load(self, **kw):
        cfg = kw.pop('config', self.cfg)
        b = pc.ExternalBindings(kw.pop('external', self.ext), **kw)
        return pc.load_biology(self.dir, cfg, bindings=b, limits=LIM)

    def test_round_trip_preserves_pending_counters_identities_and_continues_exactly(self):
        man = self.save()
        self.assertFalse(man['water_saved'])
        self.assertEqual(sorted(os.listdir(self.dir)), ['biology_arrays.npz', 'manifest.json', 'plants.csv', 'plants.metadata.json'])
        with open(os.path.join(self.dir, 'plants.csv')) as fh:
            self.assertEqual(fh.readline().strip(), 'plant_id,species,x_m,y_m,establishment_ground_m,agb_dry_kg,seasonal_agb_dry_kg,age_days,status')
        r = self.load()
        t = r.session
        self.assertEqual((t.token, t.lineage, t.counters(), t.schedule), (self.s.token, self.s.lineage, self.s.counters(), self.s.schedule))
        self.assertEqual((t.geometry_sha256, t.pop_sha256, t.bio, t.day_binding), (self.s.geometry_sha256, self.s.pop_sha256, self.s.bio,
                                                                                    self.s.day_binding))
        self.assertEqual(man['external_reference'], self.ext.to_dict())
        for k in 'DUAT':
            np.testing.assert_array_equal(t.pending()[k], self.s.pending()[k])
        self.assertIsNone(r.pending)
        u = self.s
        for i in range(3, 7):
            u, t = self.day(u, i), self.day(t, i)
        np.testing.assert_array_equal(np.asarray(u.pop.B), np.asarray(t.pop.B))
        self.assertEqual((u.counters(), t.counters()['growth_updates']), (t.counters(), 1))

    def test_new_directory_only_and_caps_before_copy(self):
        self.save()
        with self.assertRaises(pc.PlantCheckpointError):
            self.save()
        with self.assertRaises(pc.PlantCheckpointError):
            pc.save_biology(os.path.join(self.tmp, 'missing', 'ck'), self.s, external=self.ext, limits=LIM)
        with self.assertRaises(pc.PlantCheckpointError):                              # file cap: refused before any copy
            pc.save_biology(os.path.join(self.tmp, 'ck2'), self.s, external=self.ext, limits=pc.CheckpointLimits(64))
        self.assertFalse(os.path.lexists(os.path.join(self.tmp, 'ck2')))
        tight = dataclasses.replace(self.s, config=dataclasses.replace(self.s.config, max_bytes=self.s.resident_bytes() + 8))
        with self.assertRaises(pc.PlantCheckpointError):                              # global ledger: refused before any copy
            pc.save_biology(os.path.join(self.tmp, 'ck3'), tight, external=self.ext, limits=LIM)
        self.save_ok = True

    def test_load_caps_and_header_validation_before_materialization(self):
        self.save()
        with self.assertRaises(pc.PlantCheckpointError):
            pc.load_biology(self.dir, self.cfg, bindings=pc.ExternalBindings(self.ext), limits=pc.CheckpointLimits(64))
        with self.assertRaises(pc.PlantCheckpointError):
            pc.load_biology(self.dir, self.cfg, bindings=pc.ExternalBindings(self.ext),
                            limits=pc.CheckpointLimits(1 << 26, current_resident_bytes=1 << 40))
        tight = dataclasses.replace(self.cfg, max_bytes=1000)
        with self.assertRaises(pc.PlantCheckpointError):
            pc.load_biology(self.dir, tight, bindings=pc.ExternalBindings(self.ext), limits=LIM)

    def test_corruption_is_refused(self):
        for victim in ('biology_arrays.npz', 'plants.csv', 'manifest.json'):
            with self.subTest(victim=victim):
                shutil.rmtree(self.dir, ignore_errors=True)
                self.save()
                p = os.path.join(self.dir, victim)
                if victim == 'manifest.json':
                    doc = json.load(open(p))
                    doc['sequence'] += 1
                    json.dump(doc, open(p, 'w'))
                else:
                    raw = bytearray(open(p, 'rb').read())
                    raw[len(raw) // 2] ^= 0xFF
                    open(p, 'wb').write(bytes(raw))
                with self.assertRaises(pc.PlantCheckpointError):
                    self.load()

    def test_exact_external_pairing_catalog_stage_source_and_policy(self):
        self.save()
        other = nf.make_config(nf.make_catalog(n_layers=2))
        with self.assertRaises(pc.PlantCheckpointError):
            self.load(config=other)                                                  # catalog hash
        with self.assertRaises(pc.PlantCheckpointError):
            self.load(expected_stage_id={'x': 1})                                    # stale / duplicate replay
        for name, ext in (('descendant', dataclasses.replace(self.ext, hydro_count=self.ext.hydro_count + 1, request_sequence=self.ext.request_sequence + 1,
                                                             n_requests=self.ext.n_requests + 1)),
                          ('older', dataclasses.replace(self.ext, hydro_count=0, request_sequence=0, n_requests=0)),
                          ('foreign', dataclasses.replace(self.ext, lop_sha256='e' * 64)),
                          ('other digest', dataclasses.replace(self.ext, parent_digest='e' * 64))):
            with self.subTest(name), self.assertRaises(pc.PlantCheckpointError):
                self.load(external=ext)                                              # ANY unpaired state, descendants included
        with self.assertRaises(pc.PlantCheckpointError):
            pc.load_biology(self.dir, self.cfg, bindings=None, limits=LIM)
        ok = self.load(expected_stage_id=self.s.last_stage_id.to_dict())
        self.assertEqual(ok.session.last_stage_id, self.s.last_stage_id)

    def test_source_hash_drift_needs_a_typed_override_that_never_relaxes_water_identity(self):
        self.save()
        m = os.path.join(self.dir, 'manifest.json')
        doc = json.load(open(m))
        doc['sources']['growth'] = '0' * 64
        doc.pop('manifest_sha256')
        import hashlib
        doc['manifest_sha256'] = hashlib.sha256(pc._canon(doc)).hexdigest()
        json.dump(doc, open(m, 'w'))
        with self.assertRaises(pc.PlantCheckpointError):
            self.load()
        with self.assertRaises(pc.PlantCheckpointError):
            self.load(source_override=pc.SourceOverride(('uptake',), 'unrelated module named in the override'))
        ok = self.load(source_override=pc.SourceOverride(('growth',), 'reviewed: identical growth law, comment-only change'))
        self.assertEqual(ok.session.token, self.s.token)
        with self.assertRaises(pc.PlantCheckpointError):                             # the override does not relax the water pairing
            self.load(external=dataclasses.replace(self.ext, lop_sha256='e' * 64),
                      source_override=pc.SourceOverride(('growth',), 'reviewed: identical growth law, comment-only change'))
        with self.assertRaises(pc.PlantCheckpointError):
            pc.SourceOverride(('growth',), 'short')

    def test_pending_stage_after_debit_resumes_exactly_once_without_a_new_withdrawal(self):
        plan = self.s.prepare_daily(nf.day_ctx(self.a, 3), terrain_pet=nf.pet_field(self.s.pop.geometry.root_map.n_cells, 5.0))
        kw = nf.next_read_args(self.s)
        prop = self.s.propose_withdrawal(plan, nf.make_read(self.s, ordinal=3, run=nf.HEX_D, **kw))
        acc = nf.make_accepted(prop, 0.6)                    # SYRUP accepted a shortfall; LEAF then "fails" before staging
        man = self.save(external=acc.after, pending=pc.PendingStage(plan, prop, acc))
        self.assertIsNotNone(man['pending'])
        with self.assertRaises(pc.PlantCheckpointError):                             # the external reference must BE the post-debit state
            pc.save_biology(os.path.join(self.tmp, 'ck4'), self.s, external=acc.before, limits=LIM,
                            pending=pc.PendingStage(plan, prop, acc))
        with self.assertRaises(pc.PlantCheckpointError):                             # the pre-debit SYRUP state must refuse the resume
            self.load(external=acc.before)
        direct = self.s.publish(self.s.stage_day(prop, acc))
        r = self.load(external=acc.after)
        plan2, prop2, acc2 = pc.rebuild_pending(r.session, r.pending)
        self.assertEqual((prop2.identity, plan2.nonce), (prop.identity, plan.nonce))
        resumed = r.session.publish(r.session.stage_day(prop2, acc2))
        for k in 'DUAT':
            np.testing.assert_array_equal(resumed.pending()[k], direct.pending()[k])
        self.assertEqual((resumed.counters(), resumed.counters()['withdrawals_accepted']), (direct.counters(), 4))
        with self.assertRaises(ps.PlantSessionStateError):                           # the resumed stage cannot be replayed
            r.session.publish(r.session.stage_day(prop2, acc2))
        with self.assertRaises(pc.PlantCheckpointError):                             # a pending stage of another parent is refused at save
            pc.save_biology(os.path.join(self.tmp, 'ck2'), direct, external=acc.after, limits=LIM, pending=pc.PendingStage(plan, prop, acc))

    def test_biology_configuration_is_bound_seed_patch_overrides_mask_cadence(self):
        cat = self.cat
        a = self.a
        mk = lambda **kw: nf.start_session(nf.make_state(cat), nf.make_config(cat, **kw), a)
        base_kw = dict(rng_seed=10, patch_size_m=20.0)
        s = mk(**base_kw)
        idh = s.config.identity_sha256()                                               # part of the token body of every session
        self.assertNotEqual(idh, mk(rng_seed=20, patch_size_m=20.0).config.identity_sha256())
        self.assertNotEqual(idh, mk(rng_seed=10, patch_size_m=4.0).config.identity_sha256())
        self.assertEqual(idh, mk(rng_seed=10, patch_size_m=20.0).config.identity_sha256())
        out = os.path.join(self.tmp, 'cfg')
        pc.save_biology(out, s, external=s.resource_cursor, limits=LIM)
        man = pc.read_manifest(out)
        self.assertEqual((man['config']['rng_seed'], man['config']['patch_size_m']), (10, float(20.0).hex()))    # the ACTUAL values, not a flag
        ok = pc.load_biology(out, s.config, bindings=pc.ExternalBindings(s.resource_cursor), limits=LIM)
        self.assertEqual(ok.session.token, s.token)
        wide = dataclasses.replace(s.config, max_bytes=s.config.max_bytes // 2, max_edges=s.config.max_edges // 2)     # ceilings are adjustable
        self.assertEqual(pc.load_biology(out, wide, bindings=pc.ExternalBindings(s.resource_cursor), limits=LIM).session.token, s.token)
        mask = np.ones(120, dtype=np.bool_)
        variants = {'seed': dataclasses.replace(s.config, rng_seed=20), 'patch': dataclasses.replace(s.config, patch_size_m=4.0),
                    'no seed': dataclasses.replace(s.config, rng_seed=None), 'cadence': dataclasses.replace(s.config, cadence_days=3),
                    'overrides': dataclasses.replace(s.config, layer_fraction_overrides={999: [1.0 / cat.root_layers] * cat.root_layers}),
                    'mask': dataclasses.replace(s.config, canopy={'max_edges': 1 << 20, 'max_bytes': 1 << 27, 'valid_cell_mask': mask})}
        for name, cfg in variants.items():
            with self.subTest(name), self.assertRaises(pc.PlantCheckpointError):
                pc.load_biology(out, cfg, bindings=pc.ExternalBindings(s.resource_cursor), limits=LIM)
        with self.assertRaises(pc.PlantCheckpointError):                               # a SourceOverride never bypasses configuration identity
            pc.load_biology(out, variants['seed'], limits=LIM, bindings=pc.ExternalBindings(
                s.resource_cursor, source_override=pc.SourceOverride(('growth',), 'reviewed: identical growth law, comment-only change')))

    def test_unhandled_extra_T_cannot_be_paired_and_pending_identities_must_match(self):
        extra = dataclasses.replace(self.ext, request_sequence=self.ext.request_sequence + 1, n_requests=self.ext.n_requests + 1)
        with self.assertRaises(pc.PlantCheckpointError):                             # a physical T that LEAF never consumed
            pc.save_biology(os.path.join(self.tmp, 'x1'), self.s, external=extra, limits=LIM)
        self.save()
        with self.assertRaises(pc.PlantCheckpointError):
            self.load(external=extra)
        plan = self.s.prepare_daily(nf.day_ctx(self.a, 3), terrain_pet=nf.pet_field(self.s.pop.geometry.root_map.n_cells, 5.0))
        prop = self.s.propose_withdrawal(plan, nf.make_read(self.s, **nf.next_read_args(self.s)))
        acc = nf.make_accepted(prop)
        pend = lambda a: pc.PendingStage(plan, prop, a)
        bad = {'identity': dataclasses.replace(acc, identity='leaf:other'),
               'before': dataclasses.replace(acc, before=dataclasses.replace(acc.before, hydro_count=acc.before.hydro_count + 1)),
               'two T': dataclasses.replace(acc, after=dataclasses.replace(acc.after, request_sequence=acc.after.request_sequence + 1,
                                                                            n_requests=acc.after.n_requests + 1)),
               'clock': dataclasses.replace(acc, after=dataclasses.replace(acc.after, t_s=5.0))}
        for name, a in bad.items():
            with self.subTest(name), self.assertRaises(pc.PlantCheckpointError):
                pc.save_biology(os.path.join(self.tmp, 'x2'), self.s, external=a.after, limits=LIM, pending=pend(a))
        # a proposal whose plan is another object / a consumed plan is refused as well
        other_plan = self.s.prepare_daily(nf.day_ctx(self.a, 3), terrain_pet=5.0)
        with self.assertRaises(pc.PlantCheckpointError):
            pc.save_biology(os.path.join(self.tmp, 'x3'), self.s, external=acc.after, limits=LIM,
                            pending=pc.PendingStage(other_plan, prop, acc))

    def test_manifest_format_and_foreign_versions(self):
        self.save()
        man = pc.read_manifest(self.dir)
        self.assertEqual((man['format'], man['version']), (pc.FORMAT, 2))
        m = os.path.join(self.dir, 'manifest.json')
        doc = json.load(open(m))
        doc['version'] = 4                                   # the standalone leaf.checkpoint layout number is a different codec
        json.dump(doc, open(m, 'w'))
        with self.assertRaises(pc.PlantCheckpointError):
            pc.read_manifest(self.dir)


if __name__ == '__main__':
    unittest.main()
