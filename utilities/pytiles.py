from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Iterable, Sequence, Optional, Union
import numpy as np

# -----------------------------
# Tile partition
# -----------------------------


@dataclass(frozen=True)
class TilePartition:
    """
    A 1D tiling of the index range [0, N).
    Defined by cumulative 'edges': edges[0] = 0, edges[-1] = N, strictly increasing.
    Example: sizes [3, 2, 5] -> edges [0, 3, 5, 10]
    """

    edges: Tuple[int, ...]  # must start at 0 and end at N

    def __post_init__(self):
        if (
            len(self.edges) < 2
            or self.edges[0] != 0
            or any(
                self.edges[i] >= self.edges[i + 1] for i in range(len(self.edges) - 1)
            )
        ):
            raise ValueError("edges must be strictly increasing and start at 0.")

    @property
    def N(self) -> int:
        return self.edges[-1]

    @property
    def ntile(self) -> int:
        return len(self.edges) - 1

    def tile_slice(self, t: int) -> slice:
        return slice(self.edges[t], self.edges[t + 1])

    def tile_size(self, t: int) -> int:
        return self.edges[t + 1] - self.edges[t]

    def all_slices(self) -> Iterable[Tuple[int, slice]]:
        for t in range(self.ntile):
            yield t, self.tile_slice(t)

    @classmethod
    def from_sizes(cls, sizes: Sequence[int]) -> "TilePartition":
        if any(s <= 0 for s in sizes):
            raise ValueError("Tile sizes must be positive.")
        edges = [0]
        acc = 0
        for s in sizes:
            acc += s
            edges.append(acc)
        return cls(tuple(edges))

    @classmethod
    def uniform(cls, N: int, tile_size: int) -> "TilePartition":
        if tile_size <= 0:
            raise ValueError("tile_size must be positive.")
        q, r = divmod(N, tile_size)
        sizes = [tile_size] * q + ([r] if r else [])
        return cls.from_sizes(sizes if sizes else [N])


# -----------------------------
# Dense but tiled matrix (for locality-friendly MM)
# -----------------------------


def _is_zero_block(block: np.ndarray, tol: float) -> bool:
    # Max-norm thresholding; or change to Frobenius TBD?
    return np.max(np.abs(block)) <= tol


@dataclass
class TiledDenseMatrix:
    """
    Dense matrix stored as tiles.
    Blocks may be sparsely present; missing blocks are treated as zeros.
    """

    row_tiles: TilePartition
    col_tiles: TilePartition
    dtype: np.dtype = np.dtype(np.float64)
    blocks: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        self.dtype = np.dtype(self.dtype)
        # Validate any provided blocks
        for (br, bc), blk in list(self.blocks.items()):
            exp = (self.row_tiles.tile_size(br), self.col_tiles.tile_size(bc))
            if blk.shape != exp:
                raise ValueError(
                    f"Block {(br, bc)} shape mismatch, expected {exp}, got {blk.shape}"
                )
            if blk.dtype != self.dtype:
                self.blocks[(br, bc)] = blk.astype(self.dtype, copy=False, order="C")

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.row_tiles.N, self.col_tiles.N)

    def get_block(self, br: int, bc: int) -> np.ndarray:
        blk = self.blocks.get((br, bc))
        if blk is None:
            return np.zeros(
                (self.row_tiles.tile_size(br), self.col_tiles.tile_size(bc)),
                dtype=self.dtype,
            )
        return blk

    def set_block(self, br: int, bc: int, block: np.ndarray):
        exp = (self.row_tiles.tile_size(br), self.col_tiles.tile_size(bc))
        if block.shape != exp:
            raise ValueError(f"Expected block shape {exp}, got {block.shape}")
        self.blocks[(br, bc)] = np.array(block, dtype=self.dtype, copy=True, order="C")

    def to_dense(self) -> np.ndarray:
        out = np.zeros(self.shape, dtype=self.dtype)
        for (br, bc), blk in self.blocks.items():
            rs = self.row_tiles.tile_slice(br)
            cs = self.col_tiles.tile_slice(bc)
            out[rs, cs] = blk
        return out

    @classmethod
    def from_dense(
        cls,
        M: np.ndarray,
        row_tiles: TilePartition,
        col_tiles: TilePartition,
        dtype: Optional[np.dtype] = None,
    ) -> "TiledDenseMatrix":
        arr = np.asarray(M)
        if arr.shape != (row_tiles.N, col_tiles.N):
            raise ValueError("Dense shape doesn't match tile totals.")
        dtype = np.dtype(dtype if dtype is not None else arr.dtype)
        td = cls(row_tiles=row_tiles, col_tiles=col_tiles, dtype=dtype)
        for br, rs in row_tiles.all_slices():
            for bc, cs in col_tiles.all_slices():
                td.blocks[(br, bc)] = np.array(
                    arr[rs, cs], dtype=dtype, copy=True, order="C"
                )
        return td


# -----------------------------
# Block-sparse tiled Ltensor + MM
# -----------------------------


@dataclass
class BlockSparseLtensor:
    """
    TiledArray-like block-sparse storage for a rank-3 tensor L of shape (N_r, N, N),
    where each L[i] is block-sparse on a shared (row_tiles x col_tiles) grid.

    Only non-zero tiles are stored:
        blocks[(i, br, bc)] -> ndarray with shape
            (row_tiles.tile_size(br), col_tiles.tile_size(bc))

    Notes:
    - The tiling can be irregular (e.g., per-shell orbital group sizes).
    - row_tiles.N must equal col_tiles.N == N.
    - Zero-ness is decided with a threshold 'tol' (max-norm).
    """

    N_r: int
    row_tiles: TilePartition
    col_tiles: TilePartition
    dtype: np.dtype = np.dtype(np.float64)
    tol: float = 0.0
    # key: (i, br, bc)
    blocks: Dict[Tuple[int, int, int], np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        if self.row_tiles.N != self.col_tiles.N:
            raise ValueError("row_tiles.N must equal col_tiles.N (square L[i]).")
        if self.N_r < 0:
            raise ValueError("N_r must be non-negative.")
        self.dtype = np.dtype(self.dtype)

    # --------- Basic properties ----------
    @property
    def N(self) -> int:
        return self.row_tiles.N

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.N_r, self.N, self.N)

    # ---- basic block ops ----
    def add_block(
        self, i: int, br: int, bc: int, block: np.ndarray, assume_nonzero: bool = False
    ):
        """Insert a block (copied) if non-zero per tol or assume_nonzero=True."""
        if not (0 <= i < self.N_r):
            raise IndexError("i out of range.")
        if not (0 <= br < self.row_tiles.ntile and 0 <= bc < self.col_tiles.ntile):
            raise IndexError("block index out of range.")
        expected = (self.row_tiles.tile_size(br), self.col_tiles.tile_size(bc))
        if block.shape != expected:
            raise ValueError(
                f"Block shape mismatch at (i={i}, br={br}, bc={bc}). "
                f"Expected {expected}, got {block.shape}."
            )
        if assume_nonzero or not _is_zero_block(block, self.tol):
            self.blocks[(i, br, bc)] = np.array(
                block, dtype=self.dtype, copy=True, order="C"
            )

    def has_block(self, i: int, br: int, bc: int) -> bool:
        return (i, br, bc) in self.blocks

    def get_block(self, i: int, br: int, bc: int) -> np.ndarray:
        """Return the stored block; if absent, return a zero block view."""
        blk = self.blocks.get((i, br, bc))
        if blk is not None:
            return blk
        return np.zeros(
            (self.row_tiles.tile_size(br), self.col_tiles.tile_size(bc)),
            dtype=self.dtype,
        )

    # --------- Dense conversion ----------
    def dense_slice(self, i: int) -> np.ndarray:
        """Materialize L[i] as dense (N x N)."""
        if not (0 <= i < self.N_r):
            raise IndexError("i out of range.")
        out = np.zeros((self.N, self.N), dtype=self.dtype)
        for (ii, br, bc), blk in self.blocks.items():
            if ii != i:
                continue
            rs = self.row_tiles.tile_slice(br)
            cs = self.col_tiles.tile_slice(bc)
            out[rs, cs] = blk
        return out

    def to_dense(self) -> np.ndarray:
        """Materialize the full Ltensor as dense (N_r x N x N)."""
        out = np.zeros(self.shape, dtype=self.dtype)
        for (i, br, bc), blk in self.blocks.items():
            rs = self.row_tiles.tile_slice(br)
            cs = self.col_tiles.tile_slice(bc)
            out[i, rs, cs] = blk
        return out

    # --------- Memory / sparsity stats ----------
    def memory_stats(self) -> dict:
        elem_size = np.dtype(self.dtype).itemsize
        total_elems = self.N_r * self.N * self.N
        stored_elems = sum(blk.size for blk in self.blocks.values())
        total_blocks = self.N_r * self.row_tiles.ntile * self.col_tiles.ntile
        return {
            "N_r": self.N_r,
            "N": self.N,
            "tile_grid": (self.row_tiles.ntile, self.col_tiles.ntile),
            "stored_blocks": len(self.blocks),
            "total_possible_blocks": total_blocks,
            "block_density": (len(self.blocks) / total_blocks) if total_blocks else 0.0,
            "stored_elements": stored_elems,
            "total_elements": total_elems,
            "element_density": (stored_elems / total_elems) if total_elems else 0.0,
            "approx_bytes_stored": sum(b.nbytes for b in self.blocks.values()),
            "approx_bytes_dense": total_elems * elem_size,
            "dtype": str(self.dtype),
            "tol": self.tol,
        }

    def prune(self):
        """Remove blocks that are effectively zero per current tol (in-place)."""
        keys_to_drop = [
            k for k, blk in self.blocks.items() if _is_zero_block(blk, self.tol)
        ]
        for k in keys_to_drop:
            del self.blocks[k]

    # --------- Constructors ----------
    @classmethod
    def from_dense(
        cls,
        Ltensor: np.ndarray,
        row_tiles: TilePartition,
        col_tiles: TilePartition,
        tol: float = 0.0,
        dtype: Optional[np.dtype] = None,
    ) -> "BlockSparseLtensor":
        arr = np.asarray(Ltensor)
        if arr.ndim != 3:
            raise ValueError("Ltensor must have shape (N_r, N, N).")
        N_r, N1, N2 = arr.shape
        if N1 != row_tiles.N or N2 != col_tiles.N:
            raise ValueError(
                "Tile partitions must match Ltensor's trailing dimensions."
            )
        dtype = np.dtype(dtype if dtype is not None else arr.dtype)
        obj = cls(
            N_r=N_r, row_tiles=row_tiles, col_tiles=col_tiles, dtype=dtype, tol=tol
        )

        for i in range(N_r):
            for br, rs in row_tiles.all_slices():
                for bc, cs in col_tiles.all_slices():
                    block = arr[i, rs, cs]
                    if not _is_zero_block(block, tol):
                        obj.blocks[(i, br, bc)] = np.array(
                            block, dtype=dtype, copy=True, order="C"
                        )
        return obj

    @classmethod
    def from_dense_uniform(
        cls,
        Ltensor: np.ndarray,
        tile_size: int,
        tol: float = 0.0,
        dtype: Optional[np.dtype] = None,
    ) -> "BlockSparseLtensor":
        arr = np.asarray(Ltensor)
        if arr.ndim != 3:
            raise ValueError("Ltensor must have shape (N_r, N, N).")
        N_r, N, M = arr.shape
        if N != M:
            raise ValueError("Ltensor trailing dimensions must be square (N x N).")
        part = TilePartition.uniform(N, tile_size)
        return cls.from_dense(arr, part, part, tol=tol, dtype=dtype)

    # -----------------------------
    # Matrix–matrix multiply
    # -----------------------------

    def _result_dtype(self, other_dtype) -> np.dtype:
        return np.result_type(self.dtype, np.dtype(other_dtype))

    def matmul_slice(
        self,
        i: int,
        B: Union[np.ndarray, TiledDenseMatrix],
        side: str = "right",
        out: str = "dense",
        out_col_tiles: Optional[TilePartition] = None,
    ) -> Union[np.ndarray, TiledDenseMatrix]:
        """
        Multiply a single slice L[i] with a matrix B.

        Parameters
        ----------
        i : int
            Slice index for L[i].
        B : np.ndarray or TiledDenseMatrix
            If side == 'right': shape (N, M)  -> compute L[i] @ B
            If side == 'left' : shape (M, N)  -> compute B @ L[i]
        side : {'right','left'}
        out : {'dense','tiled'}
            If 'tiled', output is a TiledDenseMatrix with:
              - right: (row_tiles of L,  col_tiles of B if tiled else out_col_tiles or single tile)
              - left : (row_tiles of B if tiled else out_col_tiles or single tile, col_tiles of L)
        out_col_tiles : TilePartition, optional
            When out='tiled' and B is dense, controls the output tiling on the free dimension
            (columns for 'right', rows for 'left'). If None, uses a single tile.

        Returns
        -------
        ndarray or TiledDenseMatrix
        """
        if not (0 <= i < self.N_r):
            raise IndexError("i out of range.")

        # normalize B info
        if isinstance(B, TiledDenseMatrix):
            B_shape = B.shape
            B_dtype = B.dtype
        else:
            B = np.asarray(B)
            B_shape = B.shape
            B_dtype = B.dtype

        if side == "right":
            # L[i] (N x N) @ B (N x M) -> (N x M)
            if B_shape[0] != self.N:
                raise ValueError(
                    f"Right multiply mismatch: L is {self.N}x{self.N}, B is {B_shape}."
                )
            M = B_shape[1]
            out_dtype = self._result_dtype(B_dtype)

            if out == "dense":
                Y = np.zeros((self.N, M), dtype=out_dtype)
                if isinstance(B, TiledDenseMatrix):
                    # Y[rs, cs2] += A[br,bc] @ B[bc,cs2]
                    for (ii, br, bc), Ablk in self.blocks.items():
                        if ii != i:
                            continue
                        rs = self.row_tiles.tile_slice(br)
                        for c2, cs2 in B.col_tiles.all_slices():
                            Bblk = B.get_block(bc, c2)
                            if Bblk.size == 0:
                                continue
                            Y[rs, cs2] += Ablk @ Bblk
                else:
                    # Dense B
                    for (ii, br, bc), Ablk in self.blocks.items():
                        if ii != i:
                            continue
                        rs = self.row_tiles.tile_slice(br)
                        cs = self.col_tiles.tile_slice(bc)
                        Y[rs, :] += Ablk @ B[cs, :]
                return Y

            elif out == "tiled":
                out_row_tiles = self.row_tiles
                out_col_tiles_eff = (
                    B.col_tiles
                    if isinstance(B, TiledDenseMatrix)
                    else (
                        out_col_tiles
                        if out_col_tiles is not None
                        else TilePartition.from_sizes([M])
                    )
                )
                Yt = TiledDenseMatrix(out_row_tiles, out_col_tiles_eff, dtype=out_dtype)

                if isinstance(B, TiledDenseMatrix):
                    for (ii, br, bc), Ablk in self.blocks.items():
                        if ii != i:
                            continue
                        for c2 in range(out_col_tiles_eff.ntile):
                            prod = Ablk @ B.get_block(bc, c2)
                            if prod.size == 0:
                                continue
                            key = (br, c2)
                            Cblk = Yt.blocks.get(key)
                            if Cblk is None:
                                Yt.blocks[key] = prod.astype(out_dtype, copy=False)
                            else:
                                Cblk += prod
                else:
                    for (ii, br, bc), Ablk in self.blocks.items():
                        if ii != i:
                            continue
                        cs_rows = self.col_tiles.tile_slice(bc)
                        for c2, cs_cols in out_col_tiles_eff.all_slices():
                            prod = Ablk @ B[cs_rows, cs_cols]
                            key = (br, c2)
                            Cblk = Yt.blocks.get(key)
                            if Cblk is None:
                                Yt.blocks[key] = prod.astype(out_dtype, copy=False)
                            else:
                                Cblk += prod
                return Yt

            else:
                raise ValueError("out must be 'dense' or 'tiled'.")

        elif side == "left":
            # B (M x N) @ L[i] (N x N) -> (M x N)
            if B_shape[1] != self.N:
                raise ValueError(
                    f"Left multiply mismatch: L is {self.N}x{self.N}, B is {B_shape}."
                )
            M = B_shape[0]
            out_dtype = self._result_dtype(B_dtype)

            if out == "dense":
                Y = np.zeros((M, self.N), dtype=out_dtype)
                if isinstance(B, TiledDenseMatrix):
                    # require B.col_tiles aligned with L.row_tiles
                    if (
                        B.col_tiles.N != self.N
                        or B.col_tiles.edges != self.row_tiles.edges
                    ):
                        raise ValueError(
                            "For B @ L, B.col_tiles must match L.row_tiles."
                        )
                    for (ii, br, bc), Ablk in self.blocks.items():
                        if ii != i:
                            continue
                        cs = self.col_tiles.tile_slice(bc)
                        for r2, r2s in B.row_tiles.all_slices():
                            Y[r2s, cs] += B.get_block(r2, br) @ Ablk
                else:
                    for (ii, br, bc), Ablk in self.blocks.items():
                        if ii != i:
                            continue
                        rs = self.row_tiles.tile_slice(br)
                        cs = self.col_tiles.tile_slice(bc)
                        Y[:, cs] += B[:, rs] @ Ablk
                return Y

            elif out == "tiled":
                out_row_tiles_eff = (
                    B.row_tiles
                    if isinstance(B, TiledDenseMatrix)
                    else (
                        out_col_tiles
                        if out_col_tiles is not None
                        else TilePartition.from_sizes([M])
                    )
                )
                out_col_tiles_eff = self.col_tiles
                Yt = TiledDenseMatrix(
                    out_row_tiles_eff, out_col_tiles_eff, dtype=out_dtype
                )

                if isinstance(B, TiledDenseMatrix):
                    if (
                        B.col_tiles.N != self.N
                        or B.col_tiles.edges != self.row_tiles.edges
                    ):
                        raise ValueError(
                            "For B @ L, B.col_tiles must match L.row_tiles."
                        )
                    for (ii, br, bc), Ablk in self.blocks.items():
                        if ii != i:
                            continue
                        for r2 in range(out_row_tiles_eff.ntile):
                            prod = B.get_block(r2, br) @ Ablk
                            key = (r2, bc)
                            Cblk = Yt.blocks.get(key)
                            if Cblk is None:
                                Yt.blocks[key] = prod.astype(out_dtype, copy=False)
                            else:
                                Cblk += prod
                else:
                    for (ii, br, bc), Ablk in self.blocks.items():
                        if ii != i:
                            continue
                        rs = self.row_tiles.tile_slice(br)
                        for r2, r2s in out_row_tiles_eff.all_slices():
                            prod = B[r2s, rs] @ Ablk
                            key = (r2, bc)
                            Cblk = Yt.blocks.get(key)
                            if Cblk is None:
                                Yt.blocks[key] = prod.astype(out_dtype, copy=False)
                            else:
                                Cblk += prod
                return Yt

            else:
                raise ValueError("out must be 'dense' or 'tiled'.")

        else:
            raise ValueError("side must be 'right' or 'left'.")

    def matmul_all(
        self,
        B: Union[np.ndarray, TiledDenseMatrix],
        side: str = "right",
        out: str = "dense",
        out_col_tiles: Optional[TilePartition] = None,
    ):
        """
        Apply matmul_slice to every i in 0..N_r-1 and stack results.

        Returns
        -------
        - if out == 'dense':
            right: ndarray (N_r, N, M)
            left : ndarray (N_r, M, N)
        - if out == 'tiled':
            list[TiledDenseMatrix] of length N_r
        """
        # Normalize B for basic shape checks; reuse slice routine for the work
        if isinstance(B, TiledDenseMatrix):
            B_shape = B.shape
            B_dtype = B.dtype
        else:
            B = np.asarray(B)
            B_shape = B.shape
            B_dtype = B.dtype
        out_dtype = self._result_dtype(B_dtype)

        if out == "dense":
            if side == "right":
                if B_shape[0] != self.N:
                    raise ValueError("Dimension mismatch for right multiply.")
                M = B_shape[1]
                Y = np.zeros((self.N_r, self.N, M), dtype=out_dtype)
                for i in range(self.N_r):
                    Y[i] = self.matmul_slice(i, B, side=side, out="dense")
                return Y
            else:
                if B_shape[1] != self.N:
                    raise ValueError("Dimension mismatch for left multiply.")
                M = B_shape[0]
                Y = np.zeros((self.N_r, M, self.N), dtype=out_dtype)
                for i in range(self.N_r):
                    Y[i] = self.matmul_slice(i, B, side=side, out="dense")
                return Y
        else:
            # Return a list of tiled results
            return [
                self.matmul_slice(
                    i, B, side=side, out="tiled", out_col_tiles=out_col_tiles
                )
                for i in range(self.N_r)
            ]


if __name__ == "__main__":
    import time
    # Example sizes
    N_r, N = 12, 800
    tile_size = 50
    ntiles = N // tile_size
    part = TilePartition.uniform(N, tile_size)

    # Build a block-sparse L
    L_dense = np.zeros((N_r, N, N))
    for i in range(ntiles):
        L_dense[:, tile_size*i:tile_size*(i+1), tile_size*i:tile_size * (i+1)] = np.random.randn(N_r, tile_size, tile_size)  # non-zero tile
    for i in range(ntiles-1):
        L_dense[:, tile_size*i:tile_size*(i+1), tile_size*(i+1):tile_size*(i+2)] = np.random.randn(N_r, tile_size, tile_size)  # non-zero tile
        L_dense[:, tile_size*(i+1):tile_size*(i+2), tile_size*i:tile_size*(i+1)] = np.random.randn(N_r, tile_size, tile_size)  # non-zero tile

    # L_bst = BlockSparseLtensor.from_dense(L_dense, part, part, tol=1e-6)
    L_bst = BlockSparseLtensor.from_dense_uniform(
        L_dense, tile_size=16, tol=1e-6, dtype=np.float64
    )

    print(L_bst.memory_stats())
    # Access a dense slice if needed:
    L0 = L_bst.dense_slice(0)

    # ---- Multiply with a regular dense matrix B (N x M) ----
    B = np.random.randn(N, N)
    B_cols = TilePartition.uniform(N, tile_size)
    Bt = TiledDenseMatrix.from_dense(B, row_tiles=L_bst.col_tiles, col_tiles=B_cols)

    nruns = 10
    wt = np.zeros(3)
    t0 = time.time()
    Y0 = np.dot(L_dense, B)
    wt[0] += time.time() - t0
    for _ in range(nruns):
        # dense dgemm

        t0 = time.time()
        Y1 = L_bst.matmul_all(B, side="right", out="dense")  # shape (N_r, N, 5)
        wt[1] += (time.time() - t0) / nruns

        # ---- Multiply with a tiled dense matrix (keeps data in tiles) ----
        t0 = time.time()
        Y2 = L_bst.matmul_all(Bt, side="right", out="dense")  # shape (N_r, N, 5)
        wt[2] += (time.time() - t0) / nruns

    print("Y2 - Y0 = ", np.allclose(Y0, Y2, atol=1.e-6))
    print("Y1 - Y2 = ", np.allclose(Y1, Y2, atol=1.e-6))
    print(f"Wall time comparison: {wt[0]:.3f} {wt[1]:.3f}  {wt[2]:.3f}")
