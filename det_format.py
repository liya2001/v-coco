
# collect all detections of one image
det = {
    'image_id': _,
    # coordinates
    'person_box': _,
    # score of person-action
    # 26
    'action-name_agent': 1,
    # 26 * 2(could be less as some agent don't have role-objects)
    'action_name_role-name': 5,
}
# this format isn't very good
# could be better
