if __name__ == '__main__':
    from argparse import Namespace
    from typing import Dict
    import yaml
    import json
    import socket
    import threading
    import time
    import webbrowser
    import pickle as pkl
    from argparse import ArgumentParser
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    from pathlib import Path
    # from dataset_conf import get_keyword_to_category_mapping
    from via_utils import get_via_project_for_query

    parser = ArgumentParser("Mouse-free Annotation")
    parser.add_argument(
        "--p_dataset_config", '-pdc', type=str, default="/Users/noel/projects/PixelPick/datasets/configs/custom.yaml"
    )
    parser.add_argument(
        "--p_queries",
        type=str,
        default="/Users/noel/projects/PixelPick/checkpoints/camvid_deeplab_margin_sampling_5_p0.05_0/0_query/queries.pkl"
    )
    args: Namespace = parser.parse_args()

    dataset_config = yaml.safe_load(open(f"{args.p_dataset_config}", 'r'))
    args: dict = vars(args)
    args.update(dataset_config)
    args: Namespace = Namespace(**args)

    # Get img, queries
    dict_queries: Dict[str, dict] = pkl.load(open(args.p_queries, "rb"))

    # Change the image paths s.t. VIA can load images
    for k in list(dict_queries.keys()):
        v: dict = dict_queries.pop(k)
        dict_queries.update({
            k.replace(f"{args.dir_dataset}", f"datasets/{args.dataset_name}"): v
        })

    # load a key-category mapping which a user should define in the dataset_conf.py file.
    # keyword_category_mapping: dict = get_keyword_to_category_mapping(args.dataset_name)[0]
    keyword_category_mapping: dict = args.mapping
    # enable entering lowercase letter for annotation
    for k in list(keyword_category_mapping.keys()):
        keyword_category_mapping.update({k.lower(): keyword_category_mapping[k]})

    # Convert the query to a VIA project
    via_project = get_via_project_for_query(dict_queries, keyword_category_mapping)

    # Write via_project as JS file so that it gets loaded when browser loads the via html
    with open('via_debug_project.js', 'w') as f:
        f.write('_via_dp = ')
        json.dump(via_project, f, indent=2)

    # Launch a webserver
    HOST = 'localhost'
    PORT = 8001
    CWD = Path.cwd()
    TIMEOUT = 1
    SLEEP = 1
    MAX_ATTEMPTS = 5

    def wait_for_port_and_launch_via():
        attempts = 0
        while True:
            try:
                print(f'Trying to connect to {HOST}:{PORT}')
                with socket.create_connection((HOST, PORT), timeout=TIMEOUT):
                    break
            except OSError as e:
                time.sleep(SLEEP)
                attempts += 1
                if (attempts >= MAX_ATTEMPTS):
                    raise TimeoutError('Max attempts reached while trying to connect'
                                       f'to {HOST}:{PORT}') from e

        print('Connected')
        webbrowser.open(f'http://{HOST}:{PORT}/via_pixelpick_annotator.html')

    httpd = HTTPServer((HOST, PORT), SimpleHTTPRequestHandler)
    t = threading.Thread(target=wait_for_port_and_launch_via)

    try:
        # Launch a chrome browser to open the via html
        t.start()
        httpd.serve_forever()
    finally:
        print('Shutting Down')
        httpd.shutdown()
        t.join()
