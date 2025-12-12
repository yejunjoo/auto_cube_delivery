class Database:
    def __init__(self, left, middle, right):
        assert len(left) == 3 and len(middle) == 3 and len(right) == 3
        self.landmark = {'left':left,
                           'middle':middle,
                           'right':right}
        self.task_seq = None

        # zone1 - left - id7
        self.zone_info = {'left':
                              {'num': None, 'id': None},
                          'middle':
                              {'num': None, 'id': None},
                          'right':
                              {'num': None, 'id': None}}

        # red - left
        self.cube_info = {'red':None,
                         'blue':None,
                         'green':None}
        self.grasp_angle = None

    # def fill_landmark(self, direction, coordinate):
    #     assert direction in ['left', 'middle', 'right'], "Wrong Direction for Landmark"
    #     assert len(coordinate) == 3, "Wrong Coordinate for Landmark"
    #     self.landmark[direction] = coordinate
    #     return

    def fill_task_seq(self, seq_list):
        assert len(seq_list) > 0, "Too Short Task Sequence"
        self.task_seq = seq_list

    def fill_zone_num(self, dir, zone_num):
        assert dir in ['left', 'middle', 'right'], "Wrong Direction for Zone"
        assert zone_num in [1, 2, 3], "Wrong Number for Zone"
        self.zone_info[dir]['num'] = zone_num

    def fill_zone_id(self, dir, zone_id):
        assert dir in ['left', 'middle', 'right'], "Wrong Direction for Zone"
        assert False, "Fill Marker ID candidates"
        assert zone_id in [1, 6, 7], "Wrong Marker ID for Zone"
        self.zone_info[dir]['id'] = zone_id

    def fill_cube_info(self, color, direction):
        assert direction in ['left', 'middle', 'right'], "Wrong Direction for Cube"
        assert color in ['red', 'blue', 'green'], "Wrong Color for Cube"
        self.cube_info[color] = direction

    def memorize_grasp_angle(self, angle):
        self.grasp_angle = angle
    


    def to_environment_state(self):
        """
        create_robot_plan()에서 요구하는 형식으로
        environment_state 리스트를 생성
        [
            {"zone": 1, "cube_color": ...},
            {"zone": 2, "cube_color": ...},
            {"zone": 3, "cube_color": ...}
        ]
        """

        # 1) zone 번호 -> cube color 매핑 초기화
        zone_to_cube = {1: None, 2: None, 3: None}

        # 2) cube_info(color -> direction) 기반으로 zone 번호 찾기
        for color, direction in self.cube_info.items():
            if direction is None:
                continue

            # direction ('left', 'middle', 'right') → zone 번호
            zone_num = self.zone_info[direction]['num']
            zone_to_cube[zone_num] = color

        # 3) 최종 리스트 생성
        env = []
        for z in [1, 2, 3]:
            env.append({
                "zone": z,
                "cube_color": zone_to_cube[z]
            })

        return env