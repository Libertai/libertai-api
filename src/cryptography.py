import base64
import json
from typing import Any

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey


def create_signed_payload(data: dict[str, Any], private_key_b64: str) -> dict[str, str]:
    """
    Create a signed payload using the private key.
    
    Args:
        data: Dictionary containing the data to sign
        private_key_b64: Base64-encoded private key for signing
        
    Returns:
        Dictionary with base64-encoded data and signature
    """
    private_key_pem = base64.b64decode(private_key_b64.encode()).decode()
    private_key: RSAPrivateKey = serialization.load_pem_private_key(
        private_key_pem.encode(), password=None, backend=default_backend()
    )  # type: ignore

    # Serialize data to JSON
    json_data = json.dumps(data).encode()

    # Sign the data with private key
    encrypted_data = private_key.sign(
        json_data,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    )

    # Return base64 encoded data and signature
    return {
        "data": base64.b64encode(json_data).decode(), 
        "signature": base64.b64encode(encrypted_data).decode()
    }