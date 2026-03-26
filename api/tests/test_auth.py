"""
Authentication Tests
====================
Tests API key generation, validation, and rate limiting.
"""

import pytest
import hashlib
from api.models.api_key import APIKey


class TestAPIKeyGeneration:
    def test_generate_key_format(self):
        key = APIKey.generate()
        assert key.startswith("ngcm_")
        assert len(key) > 20

    def test_generate_unique_keys(self):
        keys = [APIKey.generate() for _ in range(10)]
        assert len(set(keys)) == 10, "Generated keys should be unique"

    def test_key_hash_consistency(self):
        key = "ngcm_test_key_12345"
        h1 = hashlib.sha256(key.encode()).hexdigest()
        h2 = hashlib.sha256(key.encode()).hexdigest()
        assert h1 == h2

    def test_key_hash_different_for_different_keys(self):
        k1 = APIKey.generate()
        k2 = APIKey.generate()
        h1 = hashlib.sha256(k1.encode()).hexdigest()
        h2 = hashlib.sha256(k2.encode()).hexdigest()
        assert h1 != h2
