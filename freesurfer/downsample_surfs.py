import os
from nipype import config
from nipype.pipeline.engine import Node, Workflow, MapNode
import nipype.interfaces.utility as util
import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
import nipype.interfaces.freesurfer as fs


def downsampel_surfs(subject_id,
                     working_dir,
                     freesurfer_dir,
                     ds_dir,
                     plugin_name,
                     use_n_procs):
    '''
    Workflow resamples e.g. native thickness maps to fsaverage5 space
    '''

    #####################################
    # GENERAL SETTINGS
    #####################################
    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
    fs.FSCommand.set_default_subjects_dir(freesurfer_dir)

    wf = Workflow(name='freesurfer_downsample')
    wf.base_dir = os.path.join(working_dir)

    nipype_cfg = dict(logging=dict(workflow_level='DEBUG'), execution={'stop_on_first_crash': True,
                                                                       'remove_unnecessary_outputs': True,
                                                                       'job_finished_timeout': 15})
    config.update_config(nipype_cfg)
    wf.config['execution']['crashdump_dir'] = os.path.join(working_dir, 'crash')

    ds = Node(nio.DataSink(base_directory=ds_dir), name='ds')
    ds.inputs.parameterization = False



    #####################
    # ITERATORS
    #####################
    # PARCELLATION ITERATOR
    infosource = Node(util.IdentityInterface(fields=['hemi', 'surf_measure', 'fwhm', 'target']), name='infosource')
    infosource.iterables = [('hemi', ['lh', 'rh']),
                            ('surf_measure', ['thickness', 'area']),
                            ('fwhm', [0, 5, 10, 20]),
                            ('target', ['fsaverage3', 'fsaverage4', 'fsaverage5']),
                            ]

    downsample = Node(fs.model.MRISPreproc(), name='downsample')
    downsample.inputs.subjects = [subject_id]
    wf.connect(infosource, 'target', downsample, 'target')
    wf.connect(infosource, 'hemi', downsample, 'hemi')
    wf.connect(infosource, 'surf_measure', downsample, 'surf_measure')
    wf.connect(infosource, 'fwhm', downsample, 'fwhm_source')

    rename = Node(util.Rename(format_string='%(hemi)s.%(surf_measure)s.%(target)s.%(fwhm)smm'), name='rename')
    rename.inputs.keep_ext = True
    wf.connect(infosource, 'target', rename, 'target')
    wf.connect(infosource, 'hemi', rename, 'hemi')
    wf.connect(infosource, 'surf_measure', rename, 'surf_measure')
    wf.connect(infosource, 'fwhm', rename, 'fwhm')
    wf.connect(downsample, 'out_file', rename, 'in_file')

    wf.connect(rename, 'out_file', ds, 'surfs.@surfs')





    #
    wf.write_graph(dotfilename=wf.name, graph2use='colored', format='pdf')  # 'hierarchical')
    wf.write_graph(dotfilename=wf.name, graph2use='orig', format='pdf')
    wf.write_graph(dotfilename=wf.name, graph2use='flat', format='pdf')

    if plugin_name == 'CondorDAGMan':
        wf.run(plugin=plugin_name)
    if plugin_name == 'MultiProc':
        wf.run(plugin=plugin_name, plugin_args={'n_procs': use_n_procs})
