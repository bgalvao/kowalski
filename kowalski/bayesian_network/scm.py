from causalnex.structure import StructureModel

def make_default_scm(radiomic_features=None):
    sm = StructureModel()
    if radiomic_features is None:

        edges_list = {
            'biopsy_grade': ['response'],
            'subtypes': ['response'],
            'histology': ['response'],
            'clinical_nodal_status': ['stage'],
            'stage': ['response'],
        }

    else:

        edges_list = {
            'Age': ['ovarian_status', *radiomic_features],
            'biopsy_grade': ['response'],
            'subtypes': ['response'],
            'histology': ['response'],
            'clinical_nodal_status': ['stage'],
            'stage': ['response'],
            **{radiomic_feature:['response'] for radiomic_feature in radiomic_features}
        }

    edges_list = [(k, dep) for k, v in edges_list.items() for dep in v]
    sm.add_edges_from(edges_list)
    
    return sm

