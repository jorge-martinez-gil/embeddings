"""Post-hoc compressors for embedding vectors."""

from embedopt.compression.base import CompressedSet, Compressor
from embedopt.compression.binary import BinaryQuantizeCompressor
from embedopt.compression.composed import ComposedCompressor
from embedopt.compression.float16 import Float16Compressor
from embedopt.compression.identity import IdentityCompressor
from embedopt.compression.opq import OptimizedProductQuantizeCompressor
from embedopt.compression.product import ProductQuantizeCompressor
from embedopt.compression.registry import build_compressor, spec_label
from embedopt.compression.scalar import ScalarQuantizeCompressor
from embedopt.compression.truncate import TruncateCompressor

__all__ = [
    "BinaryQuantizeCompressor",
    "CompressedSet",
    "ComposedCompressor",
    "Compressor",
    "Float16Compressor",
    "IdentityCompressor",
    "OptimizedProductQuantizeCompressor",
    "ProductQuantizeCompressor",
    "ScalarQuantizeCompressor",
    "TruncateCompressor",
    "build_compressor",
    "spec_label",
]
