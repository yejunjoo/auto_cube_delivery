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
You are responsible for creating a step-by-step task plan for a mobile manipulator robot.

Environment rules:
- There are exactly 3 zones in the corridor: zone 1, zone 2, and zone 3.
- Exactly two of these zones contain one cube placed on a box.
- One zone is empty.
- Cube colors are: red, green, blue.
- The robot starts at a separate starting point.
- The robot can carry only one cube at a time.
- Each zone can hold at most one cube.
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
You must generate an action sequence using ONLY the following action strings:
- pick(C)  : robot moves to the zone where cube color C is located and picks it up
             (moving + grasping are both included in this single action)
- place(Z) : robot moves to zone Z and places the currently held cube there
             (moving + placing are both included in this single action)

Here, C must be one of: R, G, B (for Red, Green, Blue).
Z must be one of: 1, 2, 3 (for zone 1, zone 2, zone 3).

Action validity rules:
- You can call pick(C) only if there is a cube with color C somewhere in the environment
  and that cube has not already been placed in its final target zone in your plan.
- You can call place(Z) only if the robot is currently holding exactly one cube.
- After place(Z), zone Z must become occupied by that cube.
- The robot can hold only one cube at a time.

SEQUENCE PATTERN (MUST ALWAYS HOLD):
- The action sequence MUST:
  - start with pick(...)
  - then strictly alternate: pick, place, pick, place, ...
  - have an EVEN number of actions (2, 4, or 6 actions for this task)
- You must NOT have two consecutive picks or two consecutive places.

# EXAMPLE 1 SIMPLE CASE (NO SWAP NEEDED):
Initial state:
- Zone 1: Blue cube (B)
- Zone 2: Green cube (G)
- Zone 3: Empty

Task Instruction:
"Put the blue cube in zone 3, and the green cube in zone 2."

In this case, the green cube is already at its correct target zone (zone 2),
so you only need to move the blue cube from zone 1 to zone 3.

A valid (and minimal) action sequence is:

FINAL_PLAN:
pick(B)   
place(3)  

# EXAMPLE 2 SWAP CASE (USING EMPTY ZONE AS TEMPORARY STORAGE):
Initial state:
- Zone 1: Blue cube (B)
- Zone 2: Green cube (G)
- Zone 3: Empty

Task Instruction:
"Put the blue cube in zone 2, and the green cube in zone 1."

In this situation, there is no extra cube and only one empty zone (zone 3).
To satisfy the instruction, you must:

1) First move the Blue cube to the empty zone (zone 3).
2) Then move the Green cube to the original position of Blue (zone 1).
3) Finally, move the Blue cube from the temporary zone (zone 3) to its final target zone (zone 2).

A valid action sequence is:

FINAL_PLAN:
pick(B)   
place(3)  
pick(G)   
place(1)  
pick(B)   
place(2)  

In this example:
- You used the empty zone (zone 3) as a temporary storage.
- The sequence strictly follows: pick, place, pick, place, pick, place.

# EXAMPLE 3 Simple SWAP CASE (USING EMPTY ZONE AS TEMPORARY STORAGE):
Initial state:
- Zone 1: Blue cube (B)
- Zone 2: Green cube (G)
- Zone 3: Empty

Task Instruction:
"Put the blue cube in zone 3, and the green cube in zone 1."

In this situation, the robot must use the empty zone (zone 3) as temporary storage
to free the target zone for the green cube. The correct sequence of operations is:

1) Move the Blue cube from zone 1 to the empty zone (zone 3).
2) Move the Green cube from zone 2 to zone 1.

A valid action sequence is:

FINAL_PLAN:
pick(B)   
place(3)  
pick(G)   
place(1)  

This completes the goal:
- Blue is now correctly placed at zone 3
- Green is now correctly placed at zone 1
- No further actions are required

IMPORTANT:
- Your action sequence MUST always:
  - start with pick(...)
  - alternate pick, place, pick, place, ...
  - have an even number of actions (2, 4, or 6), depending on the task.
- You must NEVER break the pick → place → pick → place pattern.

Output format (overall):
1) First, explain your reasoning step-by-step in natural language.
2) Then output ONLY:
   - the 'FINAL_PLAN:' line
   - followed by each action on its own line (e.g., 'pick(B)', 'place(3)', ...)
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

