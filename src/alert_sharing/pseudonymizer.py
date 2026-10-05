from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import ipaddress
import os
import re
from dataclasses import dataclass
from typing import Optional


PHASE15_HMAC_ENV = "PHASE15_HMAC_KEY_B64"
DEFAULT_KEY_ID = "phase15_hmac_v1"
DEFAULT_DIGEST_BYTES = 16

_TOKEN_RE = re.compile(
    r"^(src|dst|client)_[0-9a-f]{32}$"
)


class PseudonymizationError(ValueError):
    """Raised when keyed pseudonymization cannot be performed safely."""


def generate_hmac_key_bytes(
    size: int = 32,
) -> bytes:
    if size < 32:
        raise PseudonymizationError(
            "HMAC key must be at least 32 bytes."
        )

    return os.urandom(
        size
    )


def encode_hmac_key_b64(
    key: bytes,
) -> str:
    if not isinstance(
        key,
        (bytes, bytearray),
    ):
        raise PseudonymizationError(
            "HMAC key must be bytes."
        )

    if len(
        key
    ) < 32:
        raise PseudonymizationError(
            "HMAC key must be at least 32 bytes."
        )

    return base64.urlsafe_b64encode(
        bytes(
            key
        )
    ).decode(
        "ascii"
    )


def decode_hmac_key_b64(
    value: str,
) -> bytes:
    if not isinstance(
        value,
        str,
    ) or not value.strip():
        raise PseudonymizationError(
            "HMAC key environment value is empty."
        )

    try:
        key = base64.urlsafe_b64decode(
            value.strip().encode(
                "ascii"
            )
        )
    except (
        binascii.Error,
        UnicodeEncodeError,
    ) as exc:
        raise PseudonymizationError(
            "HMAC key must be valid URL-safe base64."
        ) from exc

    if len(
        key
    ) < 32:
        raise PseudonymizationError(
            "Decoded HMAC key must be at least 32 bytes."
        )

    return key


def load_hmac_key_from_env(
    env_name: str = PHASE15_HMAC_ENV,
) -> bytes:
    value = os.getenv(
        env_name
    )

    if not value:
        raise PseudonymizationError(
            f"Missing environment variable {env_name}. "
            "Do not hard-code the HMAC secret in source code."
        )

    return decode_hmac_key_b64(
        value
    )


def normalize_ip(
    value: str,
) -> str:
    if not isinstance(
        value,
        str,
    ) or not value.strip():
        raise PseudonymizationError(
            "IP identifier must be a non-empty string."
        )

    try:
        parsed = ipaddress.ip_address(
            value.strip()
        )
    except ValueError as exc:
        raise PseudonymizationError(
            f"Invalid IP address: {value!r}"
        ) from exc

    # Canonical text representation prevents equivalent textual forms
    # from generating different pseudonyms.
    return parsed.compressed


def normalize_opaque_identifier(
    value: str,
) -> str:
    if not isinstance(
        value,
        str,
    ) or not value.strip():
        raise PseudonymizationError(
            "Identifier must be a non-empty string."
        )

    normalized = value.strip()

    if len(
        normalized
    ) > 512:
        raise PseudonymizationError(
            "Identifier is too long."
        )

    return normalized


@dataclass(frozen=True)
class HMACPseudonymizer:
    """
    Phase 15B keyed pseudonymization.

    Security properties used here:
    - HMAC-SHA256 rather than plain SHA-256.
    - Secret key never appears in the token.
    - Domain separation prevents the same identifier from producing
      the same token across source/destination/client namespaces.
    - Only 16 bytes of the HMAC digest are exposed (32 hex chars).
    """

    key: bytes
    key_id: str = DEFAULT_KEY_ID
    digest_bytes: int = DEFAULT_DIGEST_BYTES

    def __post_init__(
        self,
    ) -> None:
        if not isinstance(
            self.key,
            (bytes, bytearray),
        ):
            raise PseudonymizationError(
                "HMAC key must be bytes."
            )

        if len(
            self.key
        ) < 32:
            raise PseudonymizationError(
                "HMAC key must be at least 32 bytes."
            )

        if not isinstance(
            self.key_id,
            str,
        ) or not self.key_id.strip():
            raise PseudonymizationError(
                "key_id must be a non-empty string."
            )

        if not (
            16
            <=
            int(
                self.digest_bytes
            )
            <=
            32
        ):
            raise PseudonymizationError(
                "digest_bytes must be between 16 and 32."
            )

    @classmethod
    def from_environment(
        cls,
        *,
        env_name: str = PHASE15_HMAC_ENV,
        key_id: str = DEFAULT_KEY_ID,
        digest_bytes: int = DEFAULT_DIGEST_BYTES,
    ) -> "HMACPseudonymizer":
        return cls(
            key=load_hmac_key_from_env(
                env_name
            ),
            key_id=key_id,
            digest_bytes=digest_bytes,
        )

    def _token(
        self,
        *,
        namespace: str,
        prefix: str,
        normalized_identifier: str,
    ) -> str:
        if namespace not in {
            "source_ip",
            "destination_ip",
            "client",
        }:
            raise PseudonymizationError(
                f"Unsupported HMAC namespace: {namespace}"
            )

        if prefix not in {
            "src",
            "dst",
            "client",
        }:
            raise PseudonymizationError(
                f"Unsupported token prefix: {prefix}"
            )

        message = (
            f"phase15|{self.key_id}|{namespace}|"
            f"{normalized_identifier}"
        ).encode(
            "utf-8"
        )

        digest = hmac.new(
            bytes(
                self.key
            ),
            message,
            hashlib.sha256,
        ).digest()[
            : int(
                self.digest_bytes
            )
        ]

        return (
            f"{prefix}_"
            +
            digest.hex()
        )

    def source_ip(
        self,
        raw_ip: str,
    ) -> str:
        return self._token(
            namespace="source_ip",
            prefix="src",
            normalized_identifier=normalize_ip(
                raw_ip
            ),
        )

    def destination_ip(
        self,
        raw_ip: str,
    ) -> str:
        return self._token(
            namespace="destination_ip",
            prefix="dst",
            normalized_identifier=normalize_ip(
                raw_ip
            ),
        )

    def client_identifier(
        self,
        raw_identifier: str,
    ) -> str:
        return self._token(
            namespace="client",
            prefix="client",
            normalized_identifier=normalize_opaque_identifier(
                raw_identifier
            ),
        )

    def pseudonymize_pair(
        self,
        *,
        source_ip: str,
        destination_ip: str,
    ) -> dict:
        return {
            "source_token":
                self.source_ip(
                    source_ip
                ),

            "destination_token":
                self.destination_ip(
                    destination_ip
                ),
        }


def token_looks_valid(
    token: Optional[str],
) -> bool:
    if token is None:
        return False

    # Phase 15A currently accepts 8..128 token characters.
    # Phase 15B freezes the HMAC output to 16 digest bytes => 32 hex chars.
    return bool(
        _TOKEN_RE.fullmatch(
            token
        )
    )
