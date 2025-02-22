import argparse

class Logger(object):
    '''
    Lightweight logging.
    TODO: replace with logging module

    '''
    def __init__(self, fh):
        self.log_fh = open(fh, 'w')

    def log(self, msg):
        '''
        Print to log file and stdout with a single command.

        '''
        print(msg, file=self.log_fh)
        print(msg)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='ldsc', type=str,
        help='Output filename prefix. If --out is not set, LDSC will use ldsc as the '
        'defualt output filename prefix.')
    # Basic LD Score Estimation Flags'
    parser.add_argument('--bfile', default=None, type=str,
        help='Prefix for Plink .bed/.bim/.fam file')
    parser.add_argument('--l2', default=False, action='store_true',
        help='Estimate l2. Compatible with both jackknife and non-jackknife.')
    # Filtering / Data Management for LD Score
    parser.add_argument('--extract', default=None, type=str,
        help='File with SNPs to include in LD Score estimation. '
        'The file should contain one SNP ID per row.')
    parser.add_argument('--keep', default=None, type=str,
        help='File with individuals to include in LD Score estimation. '
        'The file should contain one individual ID per row.')
    parser.add_argument('--ld-wind-snps', default=None, type=int,
        help='Specify the window size to be used for estimating LD Scores in units of '
        '# of SNPs. You can only specify one --ld-wind-* option.')
    parser.add_argument('--ld-wind-kb', default=None, type=float,
        help='Specify the window size to be used for estimating LD Scores in units of '
        'kilobase-pairs (kb). You can only specify one --ld-wind-* option.')
    parser.add_argument('--ld-wind-cm', default=None, type=float,
        help='Specify the window size to be used for estimating LD Scores in units of '
        'centiMorgans (cM). You can only specify one --ld-wind-* option.')
    parser.add_argument('--print-snps', default=None, type=str,
        help='This flag tells LDSC to only print LD Scores for the SNPs listed '
        '(one ID per row) in PRINT_SNPS. The sum r^2 will still include SNPs not in '
        'PRINT_SNPs. This is useful for reducing the number of LD Scores that have to be '
        'read into memory when estimating h2 or rg.' )
    # Fancy LD Score Estimation Flags
    parser.add_argument('--annot', default=None, type=str,
        help='Filename prefix for annotation file for partitioned LD Score estimation. '
        'LDSC will automatically append .annot or .annot.gz to the filename prefix. '
        'See docs/file_formats_ld for a definition of the .annot format.')
    parser.add_argument('--thin-annot', action='store_true', default=False,
        help='This flag says your annot files have only annotations, with no SNP, CM, CHR, BP columns.')
    parser.add_argument('--corr-mat', action="store_true")
    parser.add_argument('--save-mat', action="store_true")
    parser.add_argument('--no-print-condnum', action="store_true")
    parser.add_argument('--n-annot', default=None, type=int, help="Number of annotations if they are indexed as 0, 1, ..., n_annot")
    parser.add_argument('--cts-bin', default=None, type=str,
        help='This flag tells LDSC to compute partitioned LD Scores, where the partition '
        'is defined by cutting one or several continuous variable[s] into bins. '
        'The argument to this flag should be the name of a single file or a comma-separated '
        'list of files. The file format is two columns, with SNP IDs in the first column '
        'and the continuous variable in the second column. ')
    parser.add_argument('--cts-breaks', default=None, type=str,
        help='Use this flag to specify names for the continuous variables cut into bins '
        'with --cts-bin. For each continuous variable, specify breaks as a comma-separated '
        'list of breakpoints, and separate the breakpoints for each variable with an x. '
        'For example, if binning on MAF and distance to gene (in kb), '
        'you might set --cts-breaks 0.1,0.25,0.4x10,100,1000 ')
    parser.add_argument('--cts-names', default=None, type=str,
        help='Use this flag to specify names for the continuous variables cut into bins '
        'with --cts-bin. The argument to this flag should be a comma-separated list of '
        'names. For example, if binning on DAF and distance to gene, you might set '
        '--cts-bin DAF,DIST_TO_GENE ')
    parser.add_argument('--per-allele', default=False, action='store_true',
        help='Setting this flag causes LDSC to compute per-allele LD Scores, '
        'i.e., \ell_j := \sum_k p_k(1-p_k)r^2_{jk}, where p_k denotes the MAF '
        'of SNP j. ')
    parser.add_argument('--pq-exp', default=None, type=float,
        help='Setting this flag causes LDSC to compute LD Scores with the given scale factor, '
        'i.e., \ell_j := \sum_k (p_k(1-p_k))^a r^2_{jk}, where p_k denotes the MAF '
        'of SNP j and a is the argument to --pq-exp. ')
    parser.add_argument('--no-print-annot', default=False, action='store_true',
        help='By defualt, seting --cts-bin or --cts-bin-add causes LDSC to print '
        'the resulting annot matrix. Setting --no-print-annot tells LDSC not '
        'to print the annot matrix. ')
    parser.add_argument('--maf', default=None, type=float,
        help='Minor allele frequency lower bound. Default is MAF > 0.')
    # Basic Flags for Working with Variance Components
    parser.add_argument('--h2', default=None, type=str,
        help='Filename for a .sumstats[.gz] file for one-phenotype LD Score regression. '
        '--h2 requires at minimum also setting the --ref-ld and --w-ld flags.')
    parser.add_argument('--h2-cts', default=None, type=str,
        help='Filename for a .sumstats[.gz] file for cell-type-specific analysis. '
        '--h2-cts requires the --ref-ld-chr, --w-ld, and --ref-ld-chr-cts flags.')
    parser.add_argument('--rg', default=None, type=str,
        help='Comma-separated list of prefixes of .chisq filed for genetic correlation estimation.')
    parser.add_argument('--ref-ld', default=None, type=str,
        help='Use --ref-ld to tell LDSC which LD Scores to use as the predictors in the LD '
        'Score regression. '
        'LDSC will automatically append .l2.ldscore/.l2.ldscore.gz to the filename prefix.')
    parser.add_argument('--ref-ld-chr', default=None, type=str,
        help='Same as --ref-ld, but will automatically concatenate .l2.ldscore files split '
        'across 22 chromosomes. LDSC will automatically append .l2.ldscore/.l2.ldscore.gz '
        'to the filename prefix. If the filename prefix contains the symbol @, LDSC will '
        'replace the @ symbol with chromosome numbers. Otherwise, LDSC will append chromosome '
        'numbers to the end of the filename prefix.'
        'Example 1: --ref-ld-chr ld/ will read ld/1.l2.ldscore.gz ... ld/22.l2.ldscore.gz'
        'Example 2: --ref-ld-chr ld/@_kg will read ld/1_kg.l2.ldscore.gz ... ld/22_kg.l2.ldscore.gz')
    parser.add_argument('--w-ld', default=None, type=str,
        help='Filename prefix for file with LD Scores with sum r^2 taken over SNPs included '
        'in the regression. LDSC will automatically append .l2.ldscore/.l2.ldscore.gz.')
    parser.add_argument('--w-ld-chr', default=None, type=str,
        help='Same as --w-ld, but will read files split into 22 chromosomes in the same '
        'manner as --ref-ld-chr.')
    parser.add_argument('--overlap-annot', default=False, action='store_true',
        help='This flag informs LDSC that the partitioned LD Scores were generates using an '
        'annot matrix with overlapping categories (i.e., not all row sums equal 1), '
        'and prevents LDSC from displaying output that is meaningless with overlapping categories.')
    parser.add_argument('--print-coefficients',default=False,action='store_true',
        help='when categories are overlapping, print coefficients as well as heritabilities.')
    parser.add_argument('--frqfile', type=str,
        help='For use with --overlap-annot. Provides allele frequencies to prune to common '
        'snps if --not-M-5-50 is not set.')
    parser.add_argument('--frqfile-chr', type=str,
        help='Prefix for --frqfile files split over chromosome.')
    parser.add_argument('--no-intercept', action='store_true',
        help = 'If used with --h2, this constrains the LD Score regression intercept to equal '
        '1. If used with --rg, this constrains the LD Score regression intercepts for the h2 '
        'estimates to be one and the intercept for the genetic covariance estimate to be zero.')
    parser.add_argument('--intercept-h2', action='store', default=None,
        help = 'Intercepts for constrained-intercept single-trait LD Score regression.')
    parser.add_argument('--intercept-gencov', action='store', default=None,
        help = 'Intercepts for constrained-intercept cross-trait LD Score regression.'
        ' Must have same length as --rg. The first entry is ignored.')
    parser.add_argument('--M', default=None, type=str,
        help='# of SNPs (if you don\'t want to use the .l2.M files that came with your .l2.ldscore.gz files)')
    parser.add_argument('--two-step', default=None, type=float,
        help='Test statistic bound for use with the two-step estimator. Not compatible with --no-intercept and --constrain-intercept.')
    parser.add_argument('--chisq-max', default=None, type=float,
        help='Max chi^2.')
    parser.add_argument('--ref-ld-chr-cts', default=None, type=str,
        help='Name of a file that has a list of file name prefixes for cell-type-specific analysis.')
    parser.add_argument('--print-all-cts', action='store_true', default=False)

    # Flags for both LD Score estimation and h2/gencor estimation
    parser.add_argument('--print-cov', default=False, action='store_true',
        help='For use with --h2/--rg. This flag tells LDSC to print the '
        'covaraince matrix of the estimates.')
    parser.add_argument('--print-delete-vals', default=False, action='store_true',
        help='If this flag is set, LDSC will print the block jackknife delete-values ('
        'i.e., the regression coefficeints estimated from the data with a block removed). '
        'The delete-values are formatted as a matrix with (# of jackknife blocks) rows and '
        '(# of LD Scores) columns.')
    # Flags you should almost never use
    parser.add_argument('--chunk-size', default=50, type=int,
        help='Chunk size for LD Score calculation. Use the default.')
    parser.add_argument('--pickle', default=False, action='store_true',
        help='Store .l2.ldscore files as pickles instead of gzipped tab-delimited text.')
    parser.add_argument('--yes-really', default=False, action='store_true',
        help='Yes, I really want to compute whole-chromosome LD Score.')
    parser.add_argument('--invert-anyway', default=False, action='store_true',
        help="Force LDSC to attempt to invert ill-conditioned matrices.")
    parser.add_argument('--n-blocks', default=200, type=int,
        help='Number of block jackknife blocks.')
    parser.add_argument('--not-M-5-50', default=False, action='store_true',
        help='This flag tells LDSC to use the .l2.M file instead of the .l2.M_5_50 file.')
    parser.add_argument('--return-silly-things', default=False, action='store_true',
        help='Force ldsc to return silly genetic correlation estimates.')
    parser.add_argument('--no-check-alleles', default=False, action='store_true',
        help='For rg estimation, skip checking whether the alleles match. This check is '
        'redundant for pairs of chisq files generated using munge_sumstats.py and the '
        'same argument to the --merge-alleles flag.')
    # transform to liability scale
    parser.add_argument('--samp-prev',default=None,
        help='Sample prevalence of binary phenotype (for conversion to liability scale).')
    parser.add_argument('--pop-prev',default=None,
        help='Population prevalence of binary phenotype (for conversion to liability scale).')
    return parser