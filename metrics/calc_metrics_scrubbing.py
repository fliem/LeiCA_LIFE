def calc_local_metrics(preprocessed_data_dir,
                       subject_id,
                       parcellations_dict,
                       bp_freq_list,
                       fd_thresh,
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
    import utils as calc_metrics_utils





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

    bp_filter_infosource = Node(util.IdentityInterface(fields=['bp_freqs']), name='bp_filter_infosource')
    bp_filter_infosource.iterables = ('bp_freqs', bp_freq_list)

    selectfiles = Node(nio.SelectFiles(
        {
            'parcellation_time_series': '{subject_id}/con_mat/parcellated_time_series/bp_{bp_freqs}/{parcellation}/parcellation_time_series.npy'},
        base_directory=preprocessed_data_dir),
        name='selectfiles')
    selectfiles.inputs.subject_id = subject_id
    wf.connect(parcellation_infosource, 'parcellation', selectfiles, 'parcellation')
    wf.connect(bp_filter_infosource, 'bp_freqs', selectfiles, 'bp_freqs')

    fd_file = Node(nio.SelectFiles({'fd_p': '{subject_id}/QC/FD_P_ts'}, base_directory=preprocessed_data_dir),
                   name='fd_file')
    fd_file.inputs.subject_id = subject_id

    ##############
    ## CON MATS
    ##############
    ##############
    ## extract ts
    ##############

    get_good_trs = Node(util.Function(input_names=['fd_file', 'fd_thresh'],
                                      output_names=['good_trs', 'fd_scrubbed_file'],
                                      function=calc_metrics_utils.get_good_trs),
                        name='get_good_trs')
    wf.connect(fd_file, 'fd_p', get_good_trs, 'fd_file')
    get_good_trs.inputs.fd_thresh = fd_thresh

    parcellated_ts_scrubbed = Node(util.Function(input_names=['parcellation_time_series_file', 'good_trs'],
                                                 output_names=['parcellation_time_series_scrubbed'],
                                                 function=calc_metrics_utils.parcellation_time_series_scrubbing),
                                   name='parcellated_ts_scrubbed')

    wf.connect(selectfiles, 'parcellation_time_series', parcellated_ts_scrubbed, 'parcellation_time_series_file')
    wf.connect(get_good_trs, 'good_trs', parcellated_ts_scrubbed, 'good_trs')




    ##############
    ## get conmat
    ##############
    con_mat = Node(util.Function(input_names=['in_data', 'extraction_method'],
                                 output_names=['matrix', 'matrix_file'],
                                 function=calc_metrics_utils.calculate_connectivity_matrix),
                   name='con_mat')
    con_mat.inputs.extraction_method = 'correlation'
    wf.connect(parcellated_ts_scrubbed, 'parcellation_time_series_scrubbed', con_mat, 'in_data')


    ##############
    ## ds
    ##############

    wf.connect(get_good_trs, 'fd_scrubbed_file', ds, 'QC.@fd_scrubbed_file')
    fd_str = ('%.1f' % fd_thresh).replace('.', '_')
    wf.connect(con_mat, 'matrix_file', ds, 'con_mat.matrix_scrubbed_%s.@mat' % fd_str)

    # wf.write_graph(dotfilename=wf.name, graph2use='colored', format='pdf')  # 'hierarchical')
    # wf.write_graph(dotfilename=wf.name, graph2use='orig', format='pdf')
    # wf.write_graph(dotfilename=wf.name, graph2use='flat', format='pdf')

    if plugin_name == 'CondorDAGMan':
        wf.run(plugin=plugin_name, plugin_args={'initial_specs': 'request_memory = 1500'})
    if plugin_name == 'MultiProc':
        wf.run(plugin=plugin_name, plugin_args={'n_procs': use_n_procs})
