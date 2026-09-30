import io
import unittest

from scripts.modly_workflow_event_filter import filter_workflow_events


class WorkflowEventFilterTests(unittest.TestCase):
    def test_rocm_attention_descriptor_logs_do_not_fail_completed_workflow(self):
        source = io.StringIO(
            '{"type":"progress","percent":5,"label":"start"}\n'
            "a_grid_desc_m_ak_container_{540800, 256}\n"
            "b_grid_desc_n_bk_container_{64, 256}\n"
            "e_grid_desc_mblock_mperblock_nblock_nperblock_container_{540800, 64}\n"
            '{"type":"done","result":{}}\n'
        )
        self.assertEqual(filter_workflow_events(source, io.StringIO()), 0)

    def test_malformed_rocm_attention_descriptor_still_fails_closed(self):
        source = io.StringIO(
            '{"type":"done","result":{}}\n'
            "a_grid_desc_m_ak_container_{unknown, 256}\n"
        )
        self.assertEqual(filter_workflow_events(source, io.StringIO()), 1)

    def test_pinned_geo_sam2_logs_do_not_fail_a_completed_jsonl_workflow(self):
        source = io.StringIO(
            '{"type":"progress","percent":5,"label":"Validating mesh and topology revision"}\n'
            "Using device: cuda\n"
            "Found 34 objects\n"
            "Removing 0 small components\n"
            "Smoothing labels\n"
            "Split 0.0 component(s) into unique labels\n"
            "Exported labelled mesh to /workspace/StructuredAssets/runs/"
            "f6e25070-3ee3-42a5-9bf8-c98d72fd9401/geosam2-inference/upstream-output/"
            "segmentation_postprocessed_autoView01_fromPrompt00_pa0.02.glb\n"
            '{"type":"progress","percent":100,"label":"Native 3D part segmentation complete"}\n'
            '{"type":"done","result":{}}\n'
        )
        output = io.StringIO()

        self.assertEqual(filter_workflow_events(source, output), 0)
        self.assertEqual(output.getvalue(), source.getvalue())

    def test_unrecognized_plain_text_still_fails_closed(self):
        source = io.StringIO('{"type":"done","result":{}}\nUnexpected output\n')

        self.assertEqual(filter_workflow_events(source, io.StringIO()), 1)

    def test_error_event_still_fails_even_when_done_is_also_emitted(self):
        source = io.StringIO(
            '{"type":"error","code":"PART_SEGMENTATION_FAILED",'
            '"stage_id":"reference-part-segmentation","message":"failed"}\n'
            '{"type":"done","result":{}}\n'
        )

        self.assertEqual(filter_workflow_events(source, io.StringIO()), 1)

    def test_missing_done_event_fails(self):
        source = io.StringIO(
            '{"type":"progress","percent":100,"label":"complete"}\n'
        )

        self.assertEqual(filter_workflow_events(source, io.StringIO()), 1)

    def test_unknown_json_event_type_fails(self):
        source = io.StringIO('{"type":"ready"}\n{"type":"done","result":{}}\n')

        self.assertEqual(filter_workflow_events(source, io.StringIO()), 1)

    def test_known_plain_output_requires_exact_whitespace(self):
        for line in (" Found 2 objects\n", "Found 2 objects \n", "Smoothing labels \n"):
            with self.subTest(line=line):
                source = io.StringIO(line + '{"type":"done","result":{}}\n')
                self.assertEqual(filter_workflow_events(source, io.StringIO()), 1)

    def test_progress_domain_is_checked(self):
        for event in (
            '{"type":"progress","percent":101,"label":"too high"}',
            '{"type":"progress","percent":true,"label":"boolean is not an integer"}',
            '{"type":"progress","percent":5,"label":""}',
        ):
            with self.subTest(event=event):
                source = io.StringIO(event + '\n{"type":"done","result":{}}\n')
                self.assertEqual(filter_workflow_events(source, io.StringIO()), 1)

    def test_done_requires_processor_result_object(self):
        for event in ('{"type":"done"}', '{"type":"done","result":null}'):
            with self.subTest(event=event):
                self.assertEqual(filter_workflow_events(io.StringIO(event + "\n"), io.StringIO()), 1)
