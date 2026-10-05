from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from ipaddress import ip_address
from pathlib import Path
from typing import Optional

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


class TLSMaterialError(RuntimeError):
    """Raised when Phase15E2 test TLS material cannot be generated."""


@dataclass(frozen=True)
class TLSMaterial:
    ca_cert: Path
    ca_key: Path
    server_cert: Path
    server_key: Path
    client_cert: Path
    client_key: Path


def _write_private_key(
    path: Path,
    key,
) -> None:
    path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )


def _write_certificate(
    path: Path,
    cert: x509.Certificate,
) -> None:
    path.write_bytes(
        cert.public_bytes(
            serialization.Encoding.PEM
        )
    )


def _new_rsa_key():
    return rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )


def generate_local_mtls_material(
    directory: Path,
    *,
    server_dns_name: str = "localhost",
    server_ip: str = "127.0.0.1",
    client_common_name: str = "phase15_client",
    valid_days: int = 2,
) -> TLSMaterial:
    """
    Generate a short-lived local CA, server certificate, and client certificate.

    IMPORTANT:
    - Intended only for the Phase15E2 localhost research smoke test.
    - Private keys are unencrypted because they exist only inside a temporary
      directory created by the smoke-test script and are deleted afterward.
    - Production deployments should use organization-managed PKI/KMS/secret
      storage and certificate rotation.
    """

    directory = Path(directory)
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    now = datetime.now(timezone.utc)

    ca_key = _new_rsa_key()

    ca_subject = x509.Name(
        [
            x509.NameAttribute(
                NameOID.COMMON_NAME,
                "Phase15E2 Local Test CA",
            )
        ]
    )

    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_subject)
        .issuer_name(ca_subject)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=valid_days))
        .add_extension(
            x509.BasicConstraints(
                ca=True,
                path_length=0,
            ),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(
            private_key=ca_key,
            algorithm=hashes.SHA256(),
        )
    )

    # -----------------------------
    # Server certificate
    # -----------------------------
    server_key = _new_rsa_key()

    server_subject = x509.Name(
        [
            x509.NameAttribute(
                NameOID.COMMON_NAME,
                server_dns_name,
            )
        ]
    )

    server_cert = (
        x509.CertificateBuilder()
        .subject_name(server_subject)
        .issuer_name(ca_subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=valid_days))
        .add_extension(
            x509.BasicConstraints(
                ca=False,
                path_length=None,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName(server_dns_name),
                    x509.IPAddress(
                        ip_address(server_ip)
                    ),
                ]
            ),
            critical=False,
        )
        .add_extension(
            x509.ExtendedKeyUsage(
                [
                    ExtendedKeyUsageOID.SERVER_AUTH,
                ]
            ),
            critical=False,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(
            private_key=ca_key,
            algorithm=hashes.SHA256(),
        )
    )

    # -----------------------------
    # Client certificate
    # -----------------------------
    client_key = _new_rsa_key()

    client_subject = x509.Name(
        [
            x509.NameAttribute(
                NameOID.COMMON_NAME,
                client_common_name,
            )
        ]
    )

    client_cert = (
        x509.CertificateBuilder()
        .subject_name(client_subject)
        .issuer_name(ca_subject)
        .public_key(client_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=valid_days))
        .add_extension(
            x509.BasicConstraints(
                ca=False,
                path_length=None,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage(
                [
                    ExtendedKeyUsageOID.CLIENT_AUTH,
                ]
            ),
            critical=False,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(
            private_key=ca_key,
            algorithm=hashes.SHA256(),
        )
    )

    ca_cert_path = directory / "ca_cert.pem"
    ca_key_path = directory / "ca_key.pem"
    server_cert_path = directory / "server_cert.pem"
    server_key_path = directory / "server_key.pem"
    client_cert_path = directory / "client_cert.pem"
    client_key_path = directory / "client_key.pem"

    _write_private_key(
        ca_key_path,
        ca_key,
    )

    _write_certificate(
        ca_cert_path,
        ca_cert,
    )

    _write_private_key(
        server_key_path,
        server_key,
    )

    _write_certificate(
        server_cert_path,
        server_cert,
    )

    _write_private_key(
        client_key_path,
        client_key,
    )

    _write_certificate(
        client_cert_path,
        client_cert,
    )

    return TLSMaterial(
        ca_cert=ca_cert_path,
        ca_key=ca_key_path,
        server_cert=server_cert_path,
        server_key=server_key_path,
        client_cert=client_cert_path,
        client_key=client_key_path,
    )
