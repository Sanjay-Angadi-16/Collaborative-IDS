from __future__ import annotations

from dataclasses import dataclass
import ssl
import time
from typing import Any, Dict, Mapping, Optional, Union
from urllib.parse import urlparse

import httpx

from .alert_schema import PrivacyAwareAlert
from .alert_serializer import serialize_alert, serialized_sha256


class AlertSendError(RuntimeError):
    """Raised when a privacy-safe alert cannot be delivered."""


@dataclass(frozen=True)
class AlertSendResult:
    status_code: int
    response_json: Dict[str, Any]
    round_trip_ms: float
    message_sha256: str
    bytes_sent: int


def _is_localhost_url(
    url: str,
) -> bool:
    parsed = urlparse(
        url
    )

    host = (
        parsed.hostname
        or
        ""
    ).lower()

    return host in {
        "127.0.0.1",
        "localhost",
        "::1",
    }


def build_mtls_client_context(
    *,
    ca_cert_file: str,
    client_cert_file: str,
    client_key_file: str,
) -> ssl.SSLContext:
    """
    Build an SSLContext that:
    - verifies the collector certificate against the supplied CA;
    - presents the Phase15 client certificate for mutual TLS.
    """
    context = ssl.create_default_context(
        ssl.Purpose.SERVER_AUTH,
        cafile=ca_cert_file,
    )

    context.minimum_version = ssl.TLSVersion.TLSv1_2

    context.check_hostname = True

    context.load_cert_chain(
        certfile=client_cert_file,
        keyfile=client_key_file,
    )

    return context


def send_privacy_safe_alert(
    url: str,
    alert: Union[
        PrivacyAwareAlert,
        Mapping[str, Any],
    ],
    *,
    timeout_seconds: float = 5.0,
    allow_plain_http_localhost: bool = False,
    ssl_context: Optional[ssl.SSLContext] = None,
    extra_headers: Optional[Mapping[str, str]] = None,
) -> AlertSendResult:
    """
    Send one already privacy-safe Phase15 alert.

    Security policy
    ---------------
    - HTTPS is required by default.
    - Phase15E1 may explicitly allow plain HTTP ONLY for localhost.
    - Phase15E2 can pass an SSLContext containing CA trust plus
      a client certificate for mTLS.
    - X-Message-SHA256 provides integrity/tamper detection only.
      It is NOT authentication by itself.
    """

    parsed = urlparse(
        url
    )

    if parsed.scheme not in {
        "http",
        "https",
    }:
        raise AlertSendError(
            "Collector URL must use http:// or https://."
        )

    if parsed.scheme == "http":
        if not (
            allow_plain_http_localhost
            and
            _is_localhost_url(
                url
            )
        ):
            raise AlertSendError(
                "Plain HTTP is prohibited except for the explicit "
                "Phase15E1 localhost smoke test."
            )

    if (
        parsed.scheme
        ==
        "https"
        and
        ssl_context
        is None
    ):
        # Default system trust remains valid for production/public PKI.
        verify_value = True
    else:
        verify_value = (
            ssl_context
            if ssl_context is not None
            else True
        )

    payload = serialize_alert(
        alert
    )

    digest = serialized_sha256(
        payload
    )

    headers = {
        "Content-Type":
            "application/json",

        "Accept":
            "application/json",

        "X-Message-SHA256":
            digest,

        "X-Phase15-Transport":
            (
                "LOCALHOST_HTTP_SMOKE_TEST"
                if parsed.scheme == "http"
                else
                "HTTPS_TLS"
            ),
    }

    if extra_headers:
        for key, value in extra_headers.items():
            headers[
                str(
                    key
                )
            ] = str(
                value
            )

    start = time.perf_counter()

    try:
        with httpx.Client(
            verify=
                verify_value,

            timeout=
                timeout_seconds,

            trust_env=
                False,
        ) as client:
            response = client.post(
                url,
                content=
                    payload,
                headers=
                    headers,
            )

    except httpx.HTTPError as exc:
        raise AlertSendError(
            f"Alert delivery failed: {exc}"
        ) from exc

    round_trip_ms = (
        time.perf_counter()
        -
        start
    ) * 1000.0

    try:
        response_json = response.json()

    except ValueError:
        response_json = {
            "raw_response":
                response.text
        }

    return AlertSendResult(
        status_code=
            int(
                response.status_code
            ),

        response_json=
            response_json,

        round_trip_ms=
            float(
                round_trip_ms
            ),

        message_sha256=
            digest,

        bytes_sent=
            len(
                payload
            ),
    )
