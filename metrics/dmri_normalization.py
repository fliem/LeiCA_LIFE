def run_dmri_normalization_wf(subjects_list,
                              working_dir,
                              ds_dir,
                              use_n_procs,
                              plugin_name,
                              file_templates,
                              in_path):
    import os
    from nipype import config
    from nipype.pipeline.engine import Node, Workflow
    import nipype.interfaces.utility as util
    import nipype.interfaces.io as nio
    from nipype.interfaces import fsl
    from nipype.interfaces.ants import ApplyTransforms

    #####################################
    # GENERAL SETTINGS
    #####################################
    wf = Workflow(name='dmri_normalization_wf')
    wf.base_dir = os.path.join(working_dir)
    wf.config['execution']['crashdump_dir'] = os.path.join(working_dir, 'crash')

    ds = Node(nio.DataSink(), name='ds')

    ds.inputs.regexp_substitutions = [
        ('_subject_id_[A0-9]*/', ''),
        ('_metric_.*dtifit__', ''),
    ]

    infosource = Node(util.IdentityInterface(fields=['subject_id', 'metric']),
                               name='infosource')
    infosource.iterables = [('subject_id', subjects_list),
                            ('metric', ['FA', 'MD', 'L1', 'L2', 'L3'])
                            ]


    def add_subject_id_to_ds_dir_fct(subject_id, ds_dir):
        import os
        out_path = os.path.join(ds_dir, subject_id)
        return out_path

    wf.connect(infosource, ('subject_id', add_subject_id_to_ds_dir_fct, ds_dir), ds, 'base_directory')


    # GET SUBJECT SPECIFIC FUNCTIONAL DATA
    selectfiles = Node(nio.SelectFiles(file_templates, base_directory=in_path), name="selectfiles")
    wf.connect(infosource, 'subject_id', selectfiles, 'subject_id')
    wf.connect(infosource, 'metric', selectfiles, 'metric')

    #####################################
    # WF
    #####################################

    # also transform to mni space
    collect_transforms = Node(interface=util.Merge(2), name='collect_transforms')
    wf.connect([(selectfiles, collect_transforms, [('FA_2_MNI_warp', 'in1'),
                                                   ('FA_2_MNI_affine', 'in2')])
                ])

    mni = Node(interface=ApplyTransforms(), name='mni')
    #wf.connect(selectfiles, 'FA', mni, 'input_image')
    wf.connect(selectfiles, 'metric', mni, 'input_image')
    mni.inputs.reference_image = fsl.Info.standard_image('FMRIB58_FA_1mm.nii.gz')
    wf.connect(collect_transforms, 'out', mni, 'transforms')

    wf.connect(mni, 'output_image', ds, 'dti_mni')

    #####################################
    # RUN WF
    #####################################
    # wf.write_graph(dotfilename=wf.name, graph2use='colored', format='pdf')  # 'hierarchical')
    # wf.write_graph(dotfilename=wf.name, graph2use='orig', format='pdf')
    # wf.write_graph(dotfilename=wf.name, graph2use='flat', format='pdf')

    if plugin_name == 'CondorDAGMan':
        wf.run(plugin=plugin_name, plugin_args={'initial_specs': 'request_memory = 1500'})
    if plugin_name == 'MultiProc':
        wf.run(plugin=plugin_name, plugin_args={'n_procs': use_n_procs})  #
