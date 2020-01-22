def calc_local_metrics(brain_mask,
                       preprocessed_data_dir,
                       subject_id,
                       parcellations_dict,
                       bp_freq_list,
                       TR,
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
    # PARCELLATION ITERATOR
    parcellation_infosource = Node(util.IdentityInterface(fields=['parcellation']), name='parcellation_infosource')
    parcellation_infosource.iterables = ('parcellation', parcellations_dict.keys())

    # BP FILTER ITERATOR
    bp_filter_infosource = Node(util.IdentityInterface(fields=['bp_freqs']), name='bp_filter_infosource')
    bp_filter_infosource.iterables = ('bp_freqs', bp_freq_list)

    selectfiles = Node(nio.SelectFiles(selectfiles_templates,
                                       base_directory=preprocessed_data_dir),
                       name='selectfiles')
    selectfiles.inputs.subject_id = subject_id



    ##############
    ## CON MATS
    ##############
    ##############
    ## extract ts
    ##############
    parcellated_ts = Node(
        util.Function(input_names=['in_data', 'parcellation_name', 'parcellations_dict', 'bp_freqs', 'tr'],
                      output_names=['parcellation_time_series', 'parcellation_time_series_file', 'masker_file'],
                      function=calc_metrics_utils.extract_parcellation_time_series),
        name='parcellated_ts')

    parcellated_ts.inputs.parcellations_dict = parcellations_dict
    parcellated_ts.inputs.tr = TR
    wf.connect(selectfiles, 'epi_MNI_fullspectrum', parcellated_ts, 'in_data')
    wf.connect(parcellation_infosource, 'parcellation', parcellated_ts, 'parcellation_name')
    wf.connect(bp_filter_infosource, 'bp_freqs', parcellated_ts, 'bp_freqs')



    ##############
    ## get conmat
    ##############
    con_mat = Node(util.Function(input_names=['in_data', 'extraction_method'],
                                 output_names=['matrix', 'matrix_file'],
                                 function=calc_metrics_utils.calculate_connectivity_matrix),
                   name='con_mat')
    con_mat.inputs.extraction_method = 'correlation'
    wf.connect(parcellated_ts, 'parcellation_time_series', con_mat, 'in_data')


    ##############
    ## ds
    ##############

    wf.connect(parcellated_ts, 'parcellation_time_series_file', ds, 'con_mat.parcellated_time_series.@parc_ts')
    wf.connect(parcellated_ts, 'masker_file', ds, 'con_mat.parcellated_time_series.@masker')
    wf.connect(con_mat, 'matrix_file', ds, 'con_mat.matrix.@mat')

    wf.write_graph(dotfilename=wf.name, graph2use='colored', format='pdf')  # 'hierarchical')
    wf.write_graph(dotfilename=wf.name, graph2use='orig', format='pdf')
    wf.write_graph(dotfilename=wf.name, graph2use='flat', format='pdf')

    if plugin_name == 'CondorDAGMan':
        wf.run(plugin=plugin_name, plugin_args={'initial_specs': 'request_memory = 1500'})
    if plugin_name == 'MultiProc':
        wf.run(plugin=plugin_name, plugin_args={'n_procs': use_n_procs})
