import os
import re
from typing import List, Dict
from google import genai

API_KEY = os.environ.get("API_KEY")
client = genai.Client(api_key=API_KEY)

### 데이터베이스에서 environment_state 가져와야함, task_instruction 받아와야 함.
def create_robot_plan(environment_state: List[Dict], task_instruction: str) -> str:
    """
    environment_state(각 zone의 cube 상태) + task_instruction(지시사항)을 기반으로
    Gemini에게 작업 절차(plan)를 생성하도록 요청
    """

    # 1) environment_state → 텍스트 변환 
    state_lines = []
    for zone_info in environment_state:
        zone_id = zone_info["zone"]
        cube = zone_info.get("cube_color")
        cube_text = cube if cube is not None else "empty"
        state_lines.append(f"- Zone {zone_id}: {cube_text}")
    environment_state_text = "\n".join(state_lines)

    # (A) 환경 규칙 설명
    environment_description = f"""
You are responsible for generating a valid and executable manipulation plan for a mobile manipulator robot.

The environment contains:
- Exactly 3 zones: Zone 1, Zone 2, Zone 3.
- There are exactly 2 cubes in the environment.
- There is exactly one cube in each of two zones, and the remaining one zone contains no cube.
- Cube colors are chosen from (Red, Green, Blue), and the two cubes always have different colors.
- For example, Environment state: “Zone 1: blue, Zone 2: empty, Zone 3: red”
""".strip()

    # (B) 현재 environment_state
    environment_info = f"""
Environment State:
{environment_state_text}
""".strip()

    # (C) Task instruction
    instruction_info = f"""
Task Instruction:
{task_instruction}
""".strip()

    # (D) Action sequence rule that must follow
    output_rule = """
Your input will always include:
1. The current environment state (e.g., “Zone 1: blue, Zone 2: empty, Zone 3: red”)
2. One random task instruction chosen from the following official LLM Random Task List:

==================================================
LLM RANDOM TASK LIST (11 TASK TYPES)
==================================================
#1. Switch the blue cube and red cube / Switch the green cube and red cube.
Goal:
- The cube originally located in X's zone must end in Y's zone.
- The cube originally located in Y's zone must end in X's zone.
- You MUST use the empty zone as temporary storage.

Required 3-step swap algorithm (always the same):
1) Move cube X to the empty zone.
      pick(X)
      place(empty_zone)
2) Move cube Y to X's original zone.
      pick(Y)
      place(X_original_zone)
3) Move cube X from the empty zone to Y's original zone.
      pick(X)
      place(Y_original_zone)

Rules:
- Never attempt a direct swap.
- Do not move any cube other than X and Y.
- This sequence always requires exactly 6 actions.

#2. Place the blue cube at the position of the red cube. / Place the red cube at the position of the green cube.
Goal:
- Cube X must end in the zone where cube Y was initially located.
- Cube Y must NOT be moved under any circumstances.

Required minimal algorithm:
1) pick(X)
2) place(Y_original_zone)

Rules:
- Only cube X is moved.
- Cube Y remains in its initial position.
- It is allowed for both cubes to stay in the same zone.
- This task always requires exactly 2 actions.

#3. Place all cubes in Zone 2. / Place all cubes in Zone 3.
Goal:
- Both cubes must end in the target zone (order does not matter).

Required algorithm:
For each cube:
    If the cube is not already in the target zone:
        pick(cube_color)
        place(target_zone)

Rules:
- Do not relocate cubes unnecessarily.

#4. Place the red cube in Zone 1 and the blue cube in Zone 2. / Place the green cube in Zone 3 and the red cube in Zone 1.
Goal:
- Each cube must be moved to the explicitly assigned target zone.

Required algorithm:
For each cube C with target zone Z:
    If C is already in Z → skip
    Otherwise:
        pick(C)
        place(Z)

Rules:
- It is allowed for multiple cubes to end in the same zone.
- If another cube is blocking a target zone, use the empty zone temporarily.

#5. If there is any cube in Zone 1, move it to Zone 3. Otherwise, move a cube from Zone 2 to Zone 1. / If there is any cube currently in Zone 2, move it to Zone 1; otherwise, move a cube from Zone 3 to Zone 2.
Goal:
- Move exactly one cube according to the condition.

Required algorithm:
If the condition zone contains a cube:
    pick(the_cube_in_condition_zone)
    place(target_zone)
Else:
    pick(a_cube_from_alternative_zone)
    place(target_zone)

Rules:
- Only one cube must be moved.
- No unnecessary movements.
- Empty zone usage is not required.

#6. Place the blue cube in Zone 2 and the red cube in Zone 2.
Goal:
- Both blue and red cubes must end in Zone 2.

Required algorithm:
For each of (Blue, Red):
    If the cube is already in Zone 2 → skip
    Otherwise:
        pick(C)
        place(2)

Rules:
- It is allowed for multiple cubes to be in Zone 2 simultaneously.
- No need to clear the zone before placing a cube.
- Do not use the empty zone unnecessarily.

==================================================

Your job:
- Interpret the environment state + the random task instruction.
- Compute the final desired cube positions.
- Then produce a VALID action sequence using ONLY the following actions:

==================================================
ACTION DEFINITIONS
==================================================
pick(C)
    - Move to the zone that currently contains cube C and pick it up.
      (Both navigation + grasp happen inside this action.)

place(Z)
    - Move to Zone Z and place the currently held cube.
      (Both navigation + placing happen inside this action.)

Where:
  C ∈ {R, G, B}
  Z ∈ {1, 2, 3}

==================================================
ACTION RULES (VERY IMPORTANT)
==================================================
- The robot can hold only one cube at a time.
- A pick must ALWAYS be followed by a place.
- You must never output two consecutive picks or two consecutive places.
- The plan must have an EVEN number of actions.

==================================================
IMPORTANT OUTPUT FORMAT
==================================================
Your response MUST contain TWO parts:

(1) Natural-language reasoning
    Explain how you interpreted the instruction,
    how you inferred the target zone for each cube,
    and why the chosen sequence of actions is valid.

(2) FINAL_PLAN:
    Then output ONLY the plan in the following strict format:

FINAL_PLAN:
pick(C)
place(Z)
pick(C)
place(Z)
...

Do NOT add anything after the final action.
Do NOT add code blocks.
Do NOT add explanations after FINAL_PLAN.
""".strip()


    
    # 2) Prompt 합치기
    prompt = (
        environment_description + "\n\n"
        + environment_info + "\n\n"
        + instruction_info + "\n\n"
        + output_rule
    )

    # 3) Gemini 호출
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )

    return response.text


def parse_final_plan(plan_text: str) -> List[str]:
    """
    Gemini 출력에서 FINAL_PLAN 아래의 pick()/place()만 추출
    """
    lines = plan_text.splitlines()
    actions = []
    capture = False

    pattern = re.compile(r'^(pick\([RGB]\)|place\([123]\))')

    for line in lines:
        s = line.strip()

        if s.startswith("FINAL_PLAN"):
            capture = True
            continue

        if capture:
            match = pattern.match(s)
            if match:
                actions.append(match.group(1))

    if not actions:
        raise ValueError("No actions found under FINAL_PLAN section.")

    return actions

