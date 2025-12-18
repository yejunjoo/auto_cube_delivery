# zone_cube_analyzer.py

import os
import time
import json

from google import genai
from google.genai import types
from google.genai import errors

API_KEY = os.environ["API_KEY"]
_client = genai.Client(api_key=API_KEY)


class ZoneCubeAnalyzer:
    """
    한 장의 이미지에서
    - 벽에 쓰인 zone 번호 (1,2,3)
    - 박스 위 정육면체 큐브의 색 (red/blue/green, 없으면 None)
    를 한 번에 뽑아주는 헬퍼.
    """

    def __init__(
        self,
        image_path: str = "/home/ubuntu/LLM_Planning/capture/capture.jpg",
        wait_before_capture: float = 1.0,
        model_name: str = "gemini-2.5-flash",
    ):
        self.image_path = image_path
        self.wait_before_capture = wait_before_capture
        self.model_name = model_name
        self.client = _client

    def __call__(self):
        """
        사용법:
            analyzer = ZoneCubeAnalyzer(...)
            zone_num, cube_color = analyzer()

        Returns:
            zone_num (int): 1, 2, 3 중 하나
            cube_color (str | None): 'red' / 'blue' / 'green' / None
        """
        # 카메라 노드가 최신 이미지를 저장할 시간을 약간 줌
        if self.wait_before_capture > 0:
            time.sleep(self.wait_before_capture)

        image_part = self._load_image_part()
        raw_text = self._ask_gemini(image_part)
        if not raw_text:
            return None
        zone_num, cube_color = self._parse_json(raw_text)
        return zone_num, cube_color

    # ----------------- 내부 헬퍼들 ----------------- #

    def _load_image_part(self):
        with open(self.image_path, "rb") as f:
            image_bytes = f.read()
        return types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

    def _ask_gemini(self, image_part):
        prompt = """
You are analyzing a single RGB image taken by a robot in front of a wall.

In the scene:
- On the wall, there is a text indicating the zone number. It is always one of: "Zone 1", "Zone 2", or "Zone 3".
- On a black small box below, there may be a small regular hexahedron cube on the white paper. Its color is always one of: red, blue, or green.
- Sometimes there might be NO cube.

Your job:
1. Read the zone number from the wall.
2. Detect the cube color if a cube exists on the box, otherwise set cube_color to null.

STRICT OUTPUT FORMAT (very important):
- Respond with ONLY a JSON object.
- No explanation, no extra text, no Markdown code block.
- The JSON MUST have exactly these keys:
  {
    "zone_num": 1,
    "cube_color": "red"
  }
- "zone_num" must be an integer 1, 2, or 3.
- "cube_color" must be one of "red", "blue", "green", or null if no cube is present.
        """.strip()

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[image_part, prompt],
            )
        except errors.ServerError as e:
            # 503, 500 등 구글 서버 문제 (잠시 후 재시도 필요)
            print(f"Server Error (503 etc) \n{e}")
            return False



        return response.text.strip()

    def _parse_json(self, text: str):
        # 혹시 ```json ... ``` 같이 줄 수도 있으니 방어 코드
        if text.startswith("```"):
            text = text.strip("`")
            if "\n" in text:
                first_line, rest = text.split("\n", 1)
                if first_line.strip().lower() in ("json", "javascript"):
                    text = rest.strip()

        data = json.loads(text)

        zone_num = int(data["zone_num"])
        if zone_num not in (1, 2, 3):
            raise ValueError(f"Invalid zone_num from Gemini: {zone_num}")

        cube_color = data.get("cube_color", None)
        if isinstance(cube_color, str):
            cube_color = cube_color.lower().strip()
            if cube_color not in ("red", "blue", "green"):
                cube_color = None
        else:
            cube_color = None

        return zone_num, cube_color
