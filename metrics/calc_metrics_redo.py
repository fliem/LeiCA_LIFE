def calc_local_metrics(gm_wm_mask,
                       preprocessed_data_dir,
                       subject_id,
                       selectfiles_templates,
                       working_dir,
                       ds_dir,
                       use_n_procs,
                       plugin_name):
    import os
    from nipype import config
    from nipype.pipeline.engine import Node, Workflow, MapNode
    import nipype.interfaces.utility as util
    import nipype.interfaces.io as nio
    import nipype.interfaces.fsl as fsl
    from nipype.interfaces.freesurfer.preprocess import MRIConvert

    import CPAC.alff.alff as cpac_alff
    import CPAC.reho.reho as cpac_reho
    import CPAC.utils.utils as cpac_utils

    import utils as calc_metrics_utils
    from motion import calculate_FD_P, calculate_FD_J





    #####################################
    # GENERAL SETTINGS
    #####################################
    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

    wf = Workflow(name='LeiCA_LIFE_metrics')
    wf.base_dir = os.path.join(working_dir)

    nipype_cfg = dict(logging=dict(workflow_level='DEBUG'), execution={'stop_on_first_crash': True,
                                                                       'remove_unnecessary_outputs': True,
                                                                       'job_finished_timeout': 15})
    config.update_config(nipype_cfg)
    wf.config['execution']['crashdump_dir'] = os.path.join(working_dir, 'crash')

    ds = Node(nio.DataSink(base_directory=ds_dir), name='ds')
    ds.inputs.regexp_substitutions = [('MNI_resampled_brain_mask_calc.nii.gz', 'falff.nii.gz'),
                                      ('residual_filtered_3dT.nii.gz', 'alff.nii.gz'),
                                      ('_parcellation_', ''),
                                      ('_bp_freqs_', 'bp_'),
                                      ]



    #####################
    # ITERATORS
    #####################

    selectfiles = Node(nio.SelectFiles(selectfiles_templates,
                                       base_directory=preprocessed_data_dir),
                       name='selectfiles')
    selectfiles.inputs.subject_id = subject_id



    #####################
    # CALCULATE METRICS
    #####################


    # f/ALFF_MNI Z-SCORE
    alff_z = cpac_utils.get_zscore(input_name='alff', wf_name='alff_gm_wm_z')
    alff_z.inputs.inputspec.mask_file = gm_wm_mask
    wf.connect(selectfiles, 'alff', alff_z, 'inputspec.input_file')
    wf.connect(alff_z, 'outputspec.z_score_img', ds, 'alff_gm_wm_z.@alff')

    falff_z = cpac_utils.get_zscore(input_name='falff', wf_name='falff_gm_wm_z')
    falff_z.inputs.inputspec.mask_file = gm_wm_mask
    wf.connect(selectfiles, 'falff', falff_z, 'inputspec.input_file')
    wf.connect(falff_z, 'outputspec.z_score_img', ds, 'alff_gm_wm_z.@falff')



    # VARIABILITY SCORES

    variability_z = cpac_utils.get_zscore(input_name='ts_std', wf_name='variability_gm_wm_z')
    variability_z.inputs.inputspec.mask_file = gm_wm_mask
    wf.connect(selectfiles, 'variability', variability_z, 'inputspec.input_file')
    wf.connect(variability_z, 'outputspec.z_score_img', ds, 'variability_gm_wm_z.@variability_z')



    ##############
    ## ds
    ##############

    # wf.connect(parcellated_ts, 'parcellation_time_series_file', ds, 'con_mat.parcellated_time_series.@parc_ts')
    # wf.connect(parcellated_ts, 'masker_file', ds, 'con_mat.parcellated_time_series.@masker')
    # wf.connect(con_mat, 'matrix_file', ds, 'con_mat.matrix.@mat')

    wf.write_graph(dotfilename=wf.name, graph2use='colored', format='pdf')  # 'hierarchical')
    wf.write_graph(dotfilename=wf.name, graph2use='orig', format='pdf')
    wf.write_graph(dotfilename=wf.name, graph2use='flat', format='pdf')

    if plugin_name == 'CondorDAGMan':
        wf.run(plugin=plugin_name, plugin_args={'initial_specs': 'request_memory = 1500'})
    if plugin_name == 'MultiProc':
        wf.run(plugin=plugin_name, plugin_args={'n_procs': use_n_procs})
