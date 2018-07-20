import numpy as np
import json, copy, sys, os
import copy
import vsrl_utils as vu

'''
Pick annotations from coco anns, and remove negative images
'''

VCOCO_DATA_MAPPING = {'train': 'train2014', 'val': 'train2014',
                      'trainval': 'train2014', 'test': 'val2014'}


def attach_coco(json_trainval, vcoco_imlist):
    # select images that we need
    coco_imlist = [j_i['id'] for j_i in json_trainval['images']]
    coco_imlist = np.array(coco_imlist)[:, np.newaxis]
    # v-coco train-val from coco train set
    for vcoco_id in vcoco_imlist:
        if vcoco_id in coco_imlist:
            pass
        else:
            print(vcoco_id)
        # assert vcoco_id in coco_imlist
    in_vcoco = []
    for i in range(len(coco_imlist)):
        if np.any(coco_imlist[i] == vcoco_imlist):
            in_vcoco.append(i)
    j_images = [json_trainval['images'][ind] for ind in in_vcoco]

    # select annotations that we need
    coco_imlist = [j_i['image_id'] for j_i in json_trainval['annotations']]
    coco_imlist = np.array(coco_imlist)[:, np.newaxis]
    # v-coco train-val from coco train set
    for vcoco_id in vcoco_imlist:
        assert vcoco_id in coco_imlist
    in_vcoco = []
    for i in range(len(coco_imlist)):
        if np.any(coco_imlist[i] == vcoco_imlist):
            in_vcoco.append(i)
    j_annotations = [json_trainval['annotations'][ind] for ind in in_vcoco]

    json_trainval['annotations'] = j_annotations
    json_trainval['images'] = j_images

    return json_trainval


def remove_negative(vcoco_name, vcoco_path):
    '''
    Remove negative IMAGES
    :param vcoco_name:
    :return:
    '''
    print(vcoco_name)
    vcoco = vu.load_vcoco(vcoco_name)

    vcoco_imlist = np.loadtxt(os.path.join(vcoco_path, 'splits', '%s.ids' % vcoco_name)).astype(int)

    image_ids = vcoco[0]['image_id']
    vcoco_imlist = np.sort(vcoco_imlist)
    assert np.all(vcoco_imlist == np.unique(image_ids))

    keep_inds = []
    for x in vcoco:
        assert np.all(image_ids == x['image_id'])
        keep_inds.append(x['label'])
    keep_inds = np.concatenate(keep_inds, axis=1)
    keep_inds = np.where(np.sum(keep_inds, axis=1) > 0)

    print('All images num is', image_ids.shape[0])
    print('Left images nums is', keep_inds[0].size)

    image_ids = image_ids.squeeze()
    return image_ids[keep_inds]


def pick_annotations(vcoco_mode, coco_path, vcoco_path):

    coco_json_file = '{:s}/instances_{:s}.json'.format(coco_path, VCOCO_DATA_MAPPING[vcoco_mode])
    print("Loading training annotations from %s" % (format(coco_json_file)))
    coco_json = json.load(open(coco_json_file, 'r'))

    vcoco_imlist = remove_negative('vcoco_' + vcoco_mode, vcoco_path)

    vcoco_anns_json = attach_coco(coco_json, vcoco_imlist)

    vcoco_val = os.path.join(vcoco_path, 'instances_vcoco_%s_2014.json' % vcoco_mode)
    print("Writing COCO annotations needed for V-COCO to %s." % (format(vcoco_val)))
    with open(vcoco_val, 'wt') as f:
        json.dump(vcoco_anns_json, f)


if __name__ == "__main__":
    assert (len(sys.argv) == 2), \
        'Please specify coco annotation directory.'
    coco_annotation_dir = sys.argv[1]

    this_dir = os.path.dirname(__file__)
    dir_name = os.path.join(this_dir, 'data')
    vcoco_annotation_dir = dir_name

    print("%s, %s" % (coco_annotation_dir, vcoco_annotation_dir))

    vcoco_file = ['train', 'val', 'trainval', 'test']
    for vcoco_mode in vcoco_file:
        pick_annotations(vcoco_mode, coco_annotation_dir, vcoco_annotation_dir)
