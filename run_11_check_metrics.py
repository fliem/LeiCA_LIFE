
from variables import template_dir, subjects_list, metrics_root_path, wd_root_path

from utils import check_if_wf_is_ok
import os, glob

wd = 'wd_metrics_dosenbach'
wf = 'LeiCA_LIFE_metrics'

batch_path_template = os.path.join(wd_root_path, wd,  '{subject_id}', wf, 'batch')
crash_path_template = os.path.join(wd_root_path, wd, '{subject_id}', 'crash')

print(batch_path_template, crash_path_template)

everything_ok, df = check_if_wf_is_ok(batch_path_template, crash_path_template, subjects_list)
