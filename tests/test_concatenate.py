import unittest

import pandas as pd
import pyterrier as pt


class TestConcatenate(unittest.TestCase):
    def setUp(self):
        self.topics = pt.new.queries(['one', 'two', 'three'], qid=['q1', 'q2', 'q3'])

    def results(self, rows):
        return pd.DataFrame(rows, columns=['qid', 'docno', 'score']).astype({'score': 'float64'})

    def concatenate(self, left, right):
        return (pt.Transformer.from_df(left) ^ pt.Transformer.from_df(right))(self.topics)

    def test_queries_missing_from_left(self):
        left = self.results([['q1', 'd1', 0.8], ['q1', 'd2', 0.9], ['q3', 'd1', -1.0]])
        right = self.results([
            ['q1', 'd1', 10.0], ['q1', 'd3', 8.0], ['q1', 'd4', 6.0],
            ['q2', 'd1', -2.0], ['q2', 'd2', -1.0],
        ])
        actual = self.concatenate(left, right)

        self.assertEqual(
            [('q1', 'd2'), ('q1', 'd1'), ('q1', 'd3'), ('q1', 'd4'),
             ('q2', 'd2'), ('q2', 'd1'), ('q3', 'd1')],
            list(actual[['qid', 'docno']].itertuples(index=False, name=None)),
        )
        self.assertEqual([0, 1, 2, 3, 0, 1, 0], actual['rank'].tolist())
        for expected, score in zip([0.9, 0.8, 0.7999, -1.2001, -1.0, -2.0, -1.0], actual['score']):
            self.assertAlmostEqual(expected, score)
        self.assertEqual(['qid', 'query', 'docno', 'score', 'rank'], actual.columns.tolist())
        self.assertEqual(['two', 'two'], actual.loc[actual['qid'] == 'q2', 'query'].tolist())

    def test_empty_left(self):
        right = self.results([['q1', 'd2', 5.0], ['q2', 'd1', 2.0], ['q1', 'd1', 5.0]])
        actual = self.concatenate(right.head(0), right)
        self.assertEqual(['d1', 'd2', 'd1'], actual['docno'].tolist())
        self.assertEqual(['q1', 'q1', 'q2'], actual['qid'].tolist())
        self.assertEqual([5.0, 5.0, 2.0], actual['score'].tolist())
        self.assertEqual([0, 1, 0], actual['rank'].tolist())

    def test_disjoint_queries(self):
        left = self.results([['q1', 'd1', 2.0]])
        right = self.results([['q2', 'd1', 5.0]])
        actual = self.concatenate(left, right)
        self.assertEqual(['q1', 'q2'], actual['qid'].tolist())
        self.assertEqual([2.0, 5.0], actual['score'].tolist())
        self.assertEqual([0, 0], actual['rank'].tolist())

    def test_empty_right(self):
        left = self.results([['q1', 'd1', 2.0], ['q2', 'd1', 5.0]])
        actual = self.concatenate(left, left.head(0))
        self.assertEqual(['q1', 'q2'], actual['qid'].tolist())
        self.assertEqual([2.0, 5.0], actual['score'].tolist())
        self.assertEqual([0, 0], actual['rank'].tolist())

    def test_both_empty(self):
        empty = self.results([])
        actual = self.concatenate(empty, empty)
        self.assertTrue(actual.empty)
        self.assertIn('rank', actual.columns)

    def test_filtered_primary_ranking(self):
        candidates = self.results([
            ['q1', 'd1', 5.0], ['q1', 'd2', 3.0],
            ['q2', 'd1', 3.0], ['q2', 'd2', 2.0],
        ])
        all_docs = pt.Transformer.from_df(candidates)
        primary = (all_docs >> pt.apply.generic(lambda frame: frame[frame['score'] >= 4.0])
                   >> pt.apply.doc_score(lambda row: row['score'] / 10.0))
        actual = (primary ^ all_docs)(self.topics)
        self.assertEqual(['q1', 'q1', 'q2', 'q2'], actual['qid'].tolist())
        self.assertEqual(['d1', 'd2', 'd1', 'd2'], actual['docno'].tolist())
        self.assertEqual([0, 1, 0, 1], actual['rank'].tolist())
        for expected, score in zip([0.5, 0.4999, 3.0, 2.0], actual['score']):
            self.assertAlmostEqual(expected, score)

    def test_query_order_and_batching(self):
        left = self.results([['q1', 'd1', 2.0], ['q3', 'd1', 4.0]])
        right = self.results([['q2', 'd1', 5.0], ['q1', 'd2', 3.0], ['q2', 'd2', 4.0]])
        pipe = pt.Transformer.from_df(left) ^ pt.Transformer.from_df(right)
        expected = pipe(self.topics).reset_index(drop=True)
        self.assertEqual(5, len(expected))
        reversed_pipe = pt.Transformer.from_df(left.iloc[::-1]) ^ pt.Transformer.from_df(right.iloc[::-1])
        actual = reversed_pipe(self.topics.iloc[::-1]).reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, actual)
        per_query = pd.concat([pipe(self.topics.iloc[i:i + 1]) for i in range(len(self.topics))])
        pd.testing.assert_frame_equal(expected, per_query.reset_index(drop=True))
