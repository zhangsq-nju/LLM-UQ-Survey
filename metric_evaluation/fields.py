#!/usr/bin/env python3

FIELDS={
 'max_seq_prob':[('maximum_sequence_probability','confidence')],
 'max_tok_prob':[('maximum_token_probability','confidence')],
 'mean_tok_ent':[('mean_token_entropy','uncertainty')],
 'ppl':[('perplexity','uncertainty')],
 'num_set':[('number_of_semantic_sets','uncertainty')],
 'eigv':[('sum_of_eigenvalues_of_graph_laplacian','confidence')],
 'deg':[('degree_mean','confidence')],
 'ecc':[('eccentricity_mean','uncertainty')],
 'semantic_entropy':[('semantic_entropy','uncertainty'),('semantic_entropy_uncertainty','uncertainty')],
 'semantic_density':[('semantic_density_confidence','confidence'),('semantic_density_uncertainty','uncertainty')],
 'bsdetector':[('bsdetector_uncertainty','uncertainty'),('bsdetector_confidence','confidence'),('uncertainty','uncertainty')],
 'selfcheckgpt':[('selfcheck_uncertainty','uncertainty'),('selfcheck_nli_uncertainty','uncertainty'),('selfcheck_confidence','confidence'),('uncertainty','uncertainty')],
 'conu':[('conu_uncertainty','uncertainty'),('conu_confidence','confidence'),('uncertainty','uncertainty')],
 'mars':[('mars_uncertainty','uncertainty'),('uncertainty','uncertainty')],
 'p_true':[('p_true_uncertainty','uncertainty'),('ptrue_uncertainty','uncertainty'),('p_true_confidence','confidence'),('ptrue_confidence','confidence'),('p_true','confidence'),('uncertainty','uncertainty')],
 'verbalized_probability':[('verbalized_probability_uncertainty','uncertainty'),('verbalized_confidence_0_1','confidence'),('confidence','confidence'),('uncertainty','uncertainty')],
 'cot_uq_me':[('cotuq_probas_mean_uncertainty','uncertainty'),('uncertainties.Probas-mean','uncertainty'),('responses.0.uq_metrics.probas_mean','uncertainty')],
 'cot_uq_mi':[('cotuq_probas_min_uncertainty','uncertainty'),('uncertainties.Probas-min','uncertainty'),('responses.0.uq_metrics.probas_min','uncertainty')],
 'cot_uq_sar':[('cotuq_token_sar_uncertainty','uncertainty'),('uncertainties.Token-SAR','uncertainty'),('responses.0.uq_metrics.token_sar','uncertainty')],
 'cot_uq_sp':[('cotuq_self_probing_uncertainty','uncertainty'),('uncertainties.Self-Probing','uncertainty'),('responses.0.uq_metrics.self_probing','uncertainty')],
 'mean_pmi':[('mean_pointwise_mutual_information','confidence')],
 'pmi':[('pointwise_mutual_information','confidence')],
}
FALLBACKS={'deg':['metrics.the_degree_matrix.degree_mean','metrics.The Degree Matrix.degree_mean'],
           'ecc':['metrics.eccentricity.eccentricity_mean','metrics.Eccentricity.eccentricity_mean']}
FIELDS['topo_uq']=[('uncertainty','uncertainty'),('topouq_uncertainty','uncertainty')]
ALIASES={'MaxSeqProb':'max_seq_prob','MaxTokProb':'max_tok_prob','MeanTokEnt':'mean_tok_ent','PPL':'ppl',
         'NumSet':'num_set','EigV':'eigv','Deg':'deg','Ecc':'ecc','BSDetector':'bsdetector','P(True)':'p_true',
         'Semantic Entropy':'semantic_entropy','Semantic Density':'semantic_density','SelfCheckGPT':'selfcheckgpt',
         'CONU':'conu','MARS':'mars','Verbalized Probability':'verbalized_probability',
         'CoT-UQ-ME':'cot_uq_me','CoT-UQ-MI':'cot_uq_mi','CoT-UQ-SAR':'cot_uq_sar','CoT-UQ-SP':'cot_uq_sp','Topo-UQ':'topo_uq'}

def normalize_method(value):
    import re
    normalize=lambda x:re.sub('[^a-z0-9]','',x.lower())
    aliases={normalize(k):v for k,v in ALIASES.items()}
    aliases.update({normalize(k):k for k in FIELDS})
    if normalize(value) not in aliases:raise ValueError(f'Unknown method {value}; select a known method or specify score-key and score-type')
    return aliases[normalize(value)]


def nested(row,path):
    
    if isinstance(row,dict) and path in row:return row[path]
    value=row
    for key in path.split('.'):
        if isinstance(value,dict):
            if key not in value:return None
            value=value[key]
        elif isinstance(value,list) and key.isdigit() and int(key)<len(value):value=value[int(key)]
        else:return None
    return value



for _method in ('max_seq_prob','max_tok_prob','mean_tok_ent','ppl','num_set','eigv',
                'deg','ecc','semantic_entropy','semantic_density','p_true',
                'verbalized_probability','token_sar','pmi','mean_pmi'):
    FIELDS['lm_polygraph_' + _method] = [('score','uncertainty')]
