def learning_prepare_data_wf(working_dir,
                             ds_dir,
                             template_lookup_dict,
                             behav_file,
                             qc_file,
                             in_data_name_list,
                             data_lookup_dict,
                             use_n_procs,
                             plugin_name):
    import os
    from nipype import config
    from nipype.pipeline.engine import Node, Workflow
    import nipype.interfaces.utility as util
    import nipype.interfaces.io as nio
    from prepare_data_utils import vectorize_and_aggregate
    from itertools import chain

    # ensure in_data_name_list is list of lists
    in_data_name_list = [i if type(i) == list else [i] for i in in_data_name_list]
    in_data_name_list_unique = list(set(chain.from_iterable(in_data_name_list)))


    #####################################
    # GENERAL SETTINGS
    #####################################
    wf = Workflow(name='learning_prepare_data_wf')
    wf.base_dir = os.path.join(working_dir)

    nipype_cfg = dict(logging=dict(workflow_level='DEBUG'), execution={'stop_on_first_crash': False,
                                                                       'remove_unnecessary_outputs': False,
                                                                       'job_finished_timeout': 120,
                                                                       'hash_method': 'timestamp'})
    config.update_config(nipype_cfg)
    wf.config['execution']['crashdump_dir'] = os.path.join(working_dir, 'crash')

    ds = Node(nio.DataSink(), name='ds')
    ds.inputs.base_directory = os.path.join(ds_dir, 'group_learning_prepare_data')

    ds.inputs.regexp_substitutions = [
        # ('subject_id_', ''),
        ('_parcellation_', ''),
        ('_bp_freqs_', 'bp_'),
        ('_extraction_method_', ''),
        ('_subject_id_[A0-9]*/', '')
    ]

    ds_X = Node(nio.DataSink(), name='ds_X')
    ds_X.inputs.base_directory = os.path.join(ds_dir, 'vectorized_aggregated_data')

    ds_pdf = Node(nio.DataSink(), name='ds_pdf')
    ds_pdf.inputs.base_directory = os.path.join(ds_dir, 'pdfs')
    ds_pdf.inputs.parameterization = False


    #####################################
    # SET ITERATORS
    #####################################
    # SUBJECTS ITERATOR
    in_data_name_infosource = Node(util.IdentityInterface(fields=['in_data_name']), name='in_data_name_infosource')
    in_data_name_infosource.iterables = ('in_data_name', in_data_name_list_unique)

    mulitmodal_in_data_name_infosource = Node(util.IdentityInterface(fields=['multimodal_in_data_name']),
                                              name='mulitmodal_in_data_name_infosource')
    mulitmodal_in_data_name_infosource.iterables = ('multimodal_in_data_name', in_data_name_list)



    ###############################################################################################################
    # GET SUBJECTS INFO
    # create subjects list based on selection criteria

    def create_df_fct(behav_file, qc_file):
        import pandas as pd
        import os
        df = pd.read_pickle(behav_file)
        qc = pd.read_pickle(qc_file)
        df_all = qc.join(df, how='inner')

        assert df_all.index.is_unique, 'duplicates in df index. fix before cont.'

        df_all_subjects_pickle_file = os.path.abspath('df_all.pkl')
        df_all.to_pickle(df_all_subjects_pickle_file)

        full_subjects_list = df_all.index.values

        return df_all_subjects_pickle_file, full_subjects_list

    create_df = Node(util.Function(input_names=['behav_file', 'qc_file'],
                                   output_names=['df_all_subjects_pickle_file', 'full_subjects_list'],
                                   function=create_df_fct),
                     name='create_df')
    create_df.inputs.behav_file = behav_file
    create_df.inputs.qc_file = qc_file


    ###############################################################################################################
    # CREAE FILE LIST
    # of files that will be aggregted

    def create_file_list_fct(subjects_list, in_data_name, data_lookup_dict, template_lookup_dict):
        file_list = []
        for s in subjects_list:
            file_list.append(data_lookup_dict[in_data_name]['path_str'].format(subject_id=s))

        if 'matrix_name' in data_lookup_dict[in_data_name].keys():
            matrix_name = data_lookup_dict[in_data_name]['matrix_name']
        else:
            matrix_name = None

        if 'parcellation_path' in data_lookup_dict[in_data_name].keys():
            parcellation_path = data_lookup_dict[in_data_name]['parcellation_path']
        else:
            parcellation_path = None

        if 'fwhm' in data_lookup_dict[in_data_name].keys():
            fwhm = data_lookup_dict[in_data_name]['fwhm']
            if fwhm == 0:
                fwhm = None
        else:
            fwhm = None

        if 'mask_name' in data_lookup_dict[in_data_name].keys():
            mask_path = template_lookup_dict[data_lookup_dict[in_data_name]['mask_name']]
        else:
            mask_path = None

        if 'use_diagonal' in data_lookup_dict[in_data_name].keys():
            use_diagonal = data_lookup_dict[in_data_name]['use_diagonal']
        else:
            use_diagonal = False

        if 'use_fishers_z' in data_lookup_dict[in_data_name].keys():
            use_fishers_z = data_lookup_dict[in_data_name]['use_fishers_z']
        else:
            use_fishers_z = False

        if 'df_col_names' in data_lookup_dict[in_data_name].keys():
            df_col_names = data_lookup_dict[in_data_name]['df_col_names']
        else:
            df_col_names = None

        return file_list, matrix_name, parcellation_path, fwhm, mask_path, use_diagonal, use_fishers_z, df_col_names

    create_file_list = Node(util.Function(input_names=['subjects_list',
                                                       'in_data_name',
                                                       'data_lookup_dict',
                                                       'template_lookup_dict',
                                                       ],
                                          output_names=['file_list',
                                                        'matrix_name',
                                                        'parcellation_path',
                                                        'fwhm',
                                                        'mask_path',
                                                        'use_diagonal',
                                                        'use_fishers_z',
                                                        'df_col_names'],
                                          function=create_file_list_fct),
                            name='create_file_list')
    wf.connect(create_df, 'full_subjects_list', create_file_list, 'subjects_list')
    wf.connect(in_data_name_infosource, 'in_data_name', create_file_list, 'in_data_name')
    create_file_list.inputs.data_lookup_dict = data_lookup_dict
    create_file_list.inputs.template_lookup_dict = template_lookup_dict




    ###############################################################################################################
    # VECTORIZE AND AGGREGATE SUBJECTS
    # stack single subject np arrays vertically
    vectorize_aggregate_subjects = Node(util.Function(input_names=['in_data_file_list',
                                                                   'mask_file',
                                                                   'matrix_name',
                                                                   'parcellation_path',
                                                                   'fwhm',
                                                                   'use_diagonal',
                                                                   'use_fishers_z',
                                                                   'df_file',
                                                                   'df_col_names'],
                                                      output_names=['vectorized_aggregated_file',
                                                                    'unimodal_backprojection_info_file'],
                                                      function=vectorize_and_aggregate),
                                        name='vectorize_aggregate_subjects')
    wf.connect(create_file_list, 'file_list', vectorize_aggregate_subjects, 'in_data_file_list')
    wf.connect(create_file_list, 'mask_path', vectorize_aggregate_subjects, 'mask_file')
    wf.connect(create_file_list, 'matrix_name', vectorize_aggregate_subjects, 'matrix_name')
    wf.connect(create_file_list, 'parcellation_path', vectorize_aggregate_subjects, 'parcellation_path')
    wf.connect(create_file_list, 'fwhm', vectorize_aggregate_subjects, 'fwhm')
    wf.connect(create_file_list, 'use_diagonal', vectorize_aggregate_subjects, 'use_diagonal')
    wf.connect(create_file_list, 'use_fishers_z', vectorize_aggregate_subjects, 'use_fishers_z')
    wf.connect(create_df, 'df_all_subjects_pickle_file', vectorize_aggregate_subjects, 'df_file')
    wf.connect(create_file_list, 'df_col_names', vectorize_aggregate_subjects, 'df_col_names')

    wf.connect(create_df, 'df_all_subjects_pickle_file', ds_X, 'df_all_subjects_pickle_file')
    wf.connect(vectorize_aggregate_subjects, 'vectorized_aggregated_file', ds_X, 'X_file')
    wf.connect(vectorize_aggregate_subjects, 'unimodal_backprojection_info_file', ds_X, 'unimodal_backprojection_info_file')



    #####################################
    # RUN WF
    #####################################
    wf.write_graph(dotfilename=wf.name, graph2use='colored', format='pdf')  # 'hierarchical')
    wf.write_graph(dotfilename=wf.name, graph2use='orig', format='pdf')
    wf.write_graph(dotfilename=wf.name, graph2use='flat', format='pdf')

    if plugin_name == 'CondorDAGMan':
        wf.run(plugin=plugin_name)
    if plugin_name == 'MultiProc':
        wf.run(plugin=plugin_name, plugin_args={'n_procs': use_n_procs})
