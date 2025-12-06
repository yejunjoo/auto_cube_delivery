class Database:
    def __init__(self, left, middle, right):
        assert len(left) == 3 and len(middle) == 3 and len(right) == 3
        self.landmark = {'left':left,
                           'middle':middle,
                           'right':right}
        self.task_seq = None

        # zone1 - left - id7
        self.zone_info = {'zone1':
                              {'dir': None, 'id': None},
                          'zone2':
                              {'dir': None, 'id': None},
                          'zone3':
                              {'dir': None, 'id': None}}

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

    def fill_zone_dir(self, zone_name, dir):
        assert zone_name in ['zone1', 'zone2', 'zone3'], "Wrong Name for Zone"
        assert dir in ['left', 'middle', 'right'], "Wrong Direction for Zone"
        self.zone_info[zone_name]['dir'] = dir

    def fill_zone_id(self, zone_name, id):
        assert zone_name in ['zone1', 'zone2', 'zone3'], "Wrong Name for Zone"
        self.zone_info[zone_name]['id'] = id

    def fill_cube_info(self, color, direction):
        assert direction in ['left', 'middle', 'right'], "Wrong Direction for Cube"
        assert color in ['red', 'blue', 'green'], "Wrong Color for Cube"
        self.cube_info[color] = direction

    def memorize_grasp_angle(self, angle):
        self.grasp_angle = angle