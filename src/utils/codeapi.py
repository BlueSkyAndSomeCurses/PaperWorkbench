import base64
import json
import os
import re
from typing import Any, Dict, List, Optional

import httpx
from decouple import config


class CodeAPI:
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        token: Optional[str] = None,
    ):
        """
        base_url: full endpoint for execution API. For Codapi, use https://api.codapi.org/v1/exec
        token: Bearer token (read from CODEAPI_TOKEN if not provided)
        """
        if not base_url:
            base_url = config("CODEAPI_URL", "https://api.codapi.org/v1/exec")
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)
        self._token = token or os.environ.get("CODEAPI_TOKEN")

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def run_python(
        self, code: str, inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute code using Codapi-compatible schema:
          POST {base_url}
          headers: Authorization: Bearer <token>, Content-Type: application/json
          body: { sandbox: "python", command: "run", files: { "": "<code>", ...inputs_as_files } }
        """
        files: Dict[str, Any] = {"": code}
        if inputs:
            for k, v in inputs.items():
                files[str(k)] = v if isinstance(v, str) else str(v)

        payload: Dict[str, Any] = {
            "sandbox": "python",
            "command": "run",
            "files": files,
        }
        resp = self._client.post(self.base_url, json=payload, headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def extract_images_base64(response_json: Dict[str, Any]) -> List[str]:
        images: List[str] = []
        if isinstance(response_json.get("images"), list):
            images.extend(
                [img for img in response_json["images"] if isinstance(img, str)]
            )
        elif isinstance(response_json.get("image"), str):
            images.append(response_json["image"])
        files = response_json.get("files")
        if isinstance(files, dict):
            for name, content in files.items():
                data: Optional[str] = None
                if isinstance(content, dict):
                    data = (
                        content.get("content")
                        or content.get("base64")
                        or content.get("b64")
                    )
                elif isinstance(content, str):
                    data = content
                if (
                    isinstance(name, str)
                    and name.lower().endswith((".png", ".jpg", ".jpeg", ".svg"))
                    and isinstance(data, str)
                ):
                    images.append(data)
        if not images:
            stdout = response_json.get("stdout")
            if isinstance(stdout, str) and stdout:
                pattern = (
                    r"(data:image/(?:png|jpeg|jpg|svg\+xml);base64,[A-Za-z0-9+/=]+)"
                )
                matches = re.findall(pattern, stdout)
                if matches:
                    images.extend(matches)
                else:
                    for line in stdout.splitlines():
                        line = line.strip()
                        if not (line.startswith("{") and line.endswith("}")):
                            continue
                        try:
                            obj = json.loads(line)
                            for k in ("base64", "b64", "content"):
                                v = obj.get(k)
                                if isinstance(v, str):
                                    images.append(v)
                                    break
                        except Exception:
                            continue
        return images

    @staticmethod
    def b64_to_bytes(b64_str: str) -> bytes:
        if b64_str.startswith("data:") and ";base64," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        return base64.b64decode(b64_str)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
