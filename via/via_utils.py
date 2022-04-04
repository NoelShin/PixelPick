import string
import random
import numpy as np
from datetime import datetime
from collections import namedtuple as NamedTuple


ALLOWED_CHARS = [ch for ch in string.printable if ch.isalnum()]

File = NamedTuple('File', 'fid fname type loc src')
Metadata = NamedTuple('Metadata', 'vid flg z xy av')
Project = NamedTuple('Project', 'pid rev rev_timestamp pname creator created data_format_version vid_list')


def generate_random_string_for_metadata(prefix='', len=8):
    '''
    Generate random identifiers for Metadata
    CAUTION - NOT UUID, can collide with very high amount of metadata
    (or might also happen with multiprocessing?)
    '''
    return f'{prefix}_{"".join(random.choices(ALLOWED_CHARS, k=len))}'


def get_project(vid_list):
    '''
    Base Project skeleton for VIA
    '''
    project = Project(
        pid='__VIA_PROJECT_ID__',
        rev='__VIA_PROJECT_REV_ID__',
        rev_timestamp='__VIA_PROJECT_REV_TIMESTAMP__',
        pname='Pixel Pick Annotation',
        creator='Pixel Pick',
        created=int(datetime.utcnow().timestamp() * 1000),
        data_format_version='3.1.1',
        vid_list=vid_list,
    )
    return project._asdict()


def get_config(URL="http://localhost:8001/"):
    return {
        "file":{
            "loc_prefix":{
                "1":"",
                "2": URL,
                "3":"",
                "4":""
            }
        },
        "ui":{
            "file_content_align":"center",
            "file_metadata_editor_visible": False,
            "spatial_metadata_editor_visible": True,
            "spatial_region_label_attribute_id":"1"
        }
    }


def get_attribute(options):
    mapping = { k.upper(): v for k, v in options.items()}
    return {
        "1":{
            "aname":"Class",
            "anchor_id":"FILE1_Z0_XY1",
            "type":3,
            "desc":"Segmentation classes",
            "options": mapping,
            "default_option_id":""
        }
    }


def get_file(idx, name):
    return File(
        fid=idx,
        fname=name,
        type=2,
        loc=2,
        src=name,
    )._asdict()


def get_metadata(idx, xy):
    return Metadata(
        vid=idx,
        flg=0,
        z=[],
        xy=xy,
        av={}
    )._asdict()


def get_via_base_template():
    return {
        'project': {},
        'config': {},
        'attribute': {},
        'file': {},
        'view': {},
        'metadata': {},
    }


def get_via_project_for_query(query: dict, mapping):
    '''
    Function to convert query to a VIA project

    query is a dict mapping image files to corresponding
    x, y coordinates to annotate

    root_dir is the directory where image keys are expected to
    be present
    '''

    # Number of images in the query
    num_imgs = len(query)

    template = get_via_base_template()
    vid_list = [str(x) for x in range(num_imgs)]

    template['config'] = get_config()
    template['attribute'] = get_attribute(mapping)
    template['view'] = { str(i): { 'fid_list': [i]} for i in range(num_imgs)}
    template['file'] = { str(i): get_file(i, img) for i, img in enumerate(query)}
    template['metadata'] = {
        generate_random_string_for_metadata(i): get_metadata(str(i), [1, x.item(), y.item()])
        for i, k in enumerate(query) for x, y in zip(query[k]['x_coords'], query[k]['y_coords'])
    }
    template['project'] = get_project(vid_list)

    return template