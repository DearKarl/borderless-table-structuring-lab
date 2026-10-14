"""Structural fidelity and rejection of unverified localization supervision."""
import json
import unittest
from .targets import render, table_html, table_markdown
from versions.generator_pilot.targets import parse_html


class TargetsTest(unittest.TestCase):
    def setUp(self):
        self.table = dict(rows=2, cols=2, cells=[
            dict(row=0,col=0,rowspan=1,colspan=2,text='A & <B>'),
            dict(row=1,col=0,rowspan=1,colspan=1,text=''),
            dict(row=1,col=1,rowspan=1,colspan=1,text='C')])
        self.boxes = [[0,0,1,.5],[0,.5,.5,1],[.5,.5,1,1]]

    def test_span_and_empty_cell_roundtrip(self):
        self.assertEqual(parse_html(table_html(self.table)), self.table)
        self.assertEqual(table_markdown(self.table), table_html(self.table))
        _, target = render('span_detection',self.table,self.boxes)
        self.assertEqual(json.loads(target),[dict(box=[0,0,1000,500],rows=[0,0],columns=[0,1])])

    def test_cell_membership_includes_spans(self):
        _, target = render('row_column_detection',self.table,self.boxes,
            row_boxes=[[0,0,1,.5],[0,.5,1,1]],column_boxes=[[0,0,.5,1],[.5,0,1,1]])
        x=json.loads(target)
        self.assertEqual(x['columns'][0]['cells'],[0,1])
        self.assertEqual(x['columns'][1]['cells'],[0,2])
        self.assertEqual(x['rows'][0]['cells'],[0])

    def test_unverified_line_labels_rejected(self):
        with self.assertRaisesRegex(ValueError,'Complete source-verified'):
            render('ordered_text_spotting',self.table,self.boxes)

    def test_line_ranges_and_partial_query(self):
        ann=dict(status='processed',full_table_text_verified=True,cells=[
            dict(cell_index=0,status='verified',lines=[dict(text_start=0,text_end=7,normalized_box=[.1,.1,.9,.3])]),
            dict(cell_index=1,status='verified',lines=[]),
            dict(cell_index=2,status='verified',lines=[dict(text_start=0,text_end=1,normalized_box=[.6,.6,.8,.8])])])
        _, target=render('ordered_text',self.table,self.boxes,line_annotation=ann)
        self.assertEqual(json.loads(target),['A & <B>','C'])
        with self.assertRaisesRegex(ValueError,'cuts a text line'):
            render('box_text',self.table,self.boxes,line_annotation=ann,query_box=[0,0,.5,.5])


if __name__=='__main__': unittest.main()
