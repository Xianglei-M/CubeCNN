"""Run from project root: python -m unittest discover -s test -p test_cube_pooling.py"""

import unittest

import torch

from models.cube_pooling import pool_chicube_fts


class CubePoolingTests(unittest.TestCase):
    def setUp(self):
        self.features = torch.tensor([[-3., 2.], [-1., 4.], [7., -5.]])
        self.rows = torch.tensor([0, 0, 1], dtype=torch.int32)
        self.local = torch.tensor([[0, 1, 0], [0, 1, 0], [1, 0, 1]])

    def test_values_counts_empty_cells_and_permutation(self):
        expected = {"mean": [-2., 3.], "sum": [-4., 6.], "max": [-1., 4.]}
        order = torch.tensor([2, 1, 0])
        for mode, value in expected.items():
            with self.subTest(mode=mode):
                out, counts = pool_chicube_fts(self.features, self.rows,
                                              self.local, 2, 2, mode)
                torch.testing.assert_close(out[0, 0, 1, 0], torch.tensor(value))
                torch.testing.assert_close(out[1, 1, 0, 1], self.features[2])
                self.assertEqual(counts.sum().item(), 3)
                self.assertEqual(counts[0, 0, 1, 0].item(), 2)
                self.assertTrue((out[counts == 0] == 0).all())
                shuffled, _ = pool_chicube_fts(self.features[order],
                                              self.rows[order], self.local[order],
                                              2, 2, mode)
                torch.testing.assert_close(out, shuffled)

    def test_gradients(self):
        for mode, expected in [("mean", [[.5, .5], [.5, .5], [1., 1.]]),
                               ("sum", [[1., 1.]] * 3),
                               ("max", [[0., 0.], [1., 1.], [1., 1.]])]:
            f = self.features.clone().requires_grad_()
            out, _ = pool_chicube_fts(f, self.rows, self.local, 2, 2, mode)
            out.sum().backward()
            torch.testing.assert_close(f.grad, torch.tensor(expected))

    def test_empty_input(self):
        for mode in ("mean", "sum", "max"):
            f = torch.empty(0, 2, requires_grad=True)
            out, counts = pool_chicube_fts(f, torch.empty(0, dtype=torch.long),
                                          torch.empty(0, 3, dtype=torch.long),
                                          1, 2, mode)
            self.assertTrue((out == 0).all())
            self.assertEqual(counts.sum().item(), 0)
            out.sum().backward()
            self.assertIsNotNone(f.grad)

    def test_invalid_indices(self):
        for rows, local in [(torch.tensor([0, 0, 2]), self.local),
                            (self.rows, self.local - 2)]:
            with self.assertRaises(ValueError):
                pool_chicube_fts(self.features, rows, local, 2, 2)

    def test_half_accumulates_in_float32(self):
        out, _ = pool_chicube_fts(self.features.half(), self.rows,
                                  self.local, 2, 2)
        self.assertEqual(out.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
