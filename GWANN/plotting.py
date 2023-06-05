# Experiment summary generation

def return_data(d, model_fold, label, bp, sys_params, covs, 
    train_ids_f, test_ids_f):
    geno = None
    phen_cov = [] 

    gs, model_path, Xs, ys, snps, cws, colss = [], [], [], [], [], [], [] 
    for i, gene in enumerate(d['names']):
        chrom = d['chrom'][i]
        X, y, X_test, y_test, class_weights, data_cols, num_snps = load_data(
            {chrom:geno}, phen_cov, [gene,], [chrom,], label, bp, 
            '/home/upamanyu/GWASOnSteroids/Runs', 
            sys_params, covs, train_ids_f, test_ids_f,
            over_coeff=0.0, 
            balance=1.0, 
            SNP_thresh=1000)
        # X_test, y_test = X_test[0], y_test[0]
        # X_test = np.concatenate((X_test[0], X_test[1]))
        # y_test = np.concatenate((y_test[0], y_test[1]))
        _, _, Xt, yt = group_data_prep(None, None, X_test, y_test, 10, covs)
        
        # skf = StratifiedKFold(n_splits=3, shuffle=False)
        # for train_index, test_index in skf.split(Xt, yt):
        #     Xt = Xt[test_index]
        #     yt = yt[test_index]
        #     break
        
        gs.append(gene)
        model_path.append('{}/{}/{}_{}.pt'.format(model_fold, gene, num_snps, gene))
        Xs.append(Xt)
        ys.append(yt)
        snps.append(num_snps)
        cws.append(class_weights)
        colss.append(data_cols)

    return gs, model_path, Xs, ys, snps, cws, colss

def return_stats(df, unc_p, cor_p):
    """[summary]

    Parameters
    ----------
    df : [type]
        [description]
    unc_p : [type]
        [description]
    cor_p : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    tot = df.shape[0]
    cor = df.loc[df['P_Acc'] <= cor_p].shape[0]
    unc = df.loc[df['P_Acc'] <= unc_p].shape[0]
    vprint('Corrected: {}/{} = {:.2f}'.format(cor, tot, 100*cor/tot))
    vprint('Uncorrected: {}/{} = {:.2f}\n'.format(unc, tot, 100*unc/tot))
    return tot, cor, unc

def ptest_neg_summary_stats(logs_file, cor_p):
    """[summary]

    Parameters
    ----------
    logs_file : [type]
        [description]
    cor_p : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    df = pd.read_csv(logs_file)
    df.set_index('Gene', drop=False, inplace=True)
    with open('./params/gene_subsets.yaml', 'r') as f:
        g_yaml = yaml.load(f, Loader=yaml.FullLoader)

    summary_dict = {
        'Unc_Neg':0,
        'Corr_Neg':0
    }
    
    vprint('OVERALL (NEG)')
    vprint('------------')
    t, c, u = return_stats(df, 0.05, cor_p)
    summary_dict['Unc_Neg'] = u
    summary_dict['Corr_Neg'] = c
    
    try:
        df11 = df.loc[g_yaml['First_GroupTrain_Hits_Neg']['names']]
        vprint('OLD 11 (NEG)')
        vprint('------------')
        t, c, u = return_stats(df11, 0.05, cor_p)
        summary_dict['Unc_11_Neg'] = u
        summary_dict['Corr_11_Neg'] = c
        
        dfM = df.loc[~df.index.isin(df11.index.tolist())]
        vprint('MARIONI (NEG)')
        vprint('------------')
        t, c, u = return_stats(dfM, 0.05, cor_p)
        summary_dict['Unc_Marioni_Neg'] = u
        summary_dict['Corr_Marioni_Neg'] = c
    except:
        print('Diabetes')

    return summary_dict

def ptest_pos_summary_stats(logs_file, cor_p):
    """[summary]

    Parameters
    ----------
    logs_file : [type]
        [description]
    cor_p : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    df = pd.read_csv(logs_file)
    df.set_index('Gene', drop=False, inplace=True)

    with open('./params/gene_subsets.yaml', 'r') as f:
        g_yaml = yaml.load(f, Loader=yaml.FullLoader)

    summary_dict = {
        'Unc_Pos':0,
        'Corr_Pos':0
    }
    
    vprint('OVERALL (POS)')
    vprint('------------')
    t, c, u = return_stats(df, 0.05, cor_p)
    summary_dict['Unc_Pos'] = u
    summary_dict['Corr_Pos'] = c
    try:
        df19 = df.loc[g_yaml['First_GroupTrain_Hits_Pos']['names']]
        vprint('OLD 19 (POS)')
        vprint('------------')
        t, c, u = return_stats(df19, 0.05, cor_p)
        summary_dict['Unc_19_Pos'] = u
        summary_dict['Corr_19_Pos'] = c
        
        dfM = df.loc[df.index.isin(g_yaml['Marioni_Top50']['names'])]
        vprint('MARIONI (POS)')
        vprint('------------')
        t, c, u = return_stats(dfM, 0.05, cor_p)
        summary_dict['Unc_Marioni_Pos'] = u
        summary_dict['Corr_Marioni_Pos'] = c
        
        dfK = df.loc[~df.index.isin(dfM.index.tolist() + df19.index.tolist())]
        vprint('Kegg (POS)')
        vprint('------------')
        t, c, u = return_stats(dfK, 0.05, cor_p)
        summary_dict['Unc_KEGG_Pos'] = u
        summary_dict['Corr_KEGG_Pos'] = c
    except:
        print('Diabetes')

    return summary_dict

def method_comparison(df, cor_p, fname):
    x = df.Gene.values
    y = df.columns[1:]
    data = df.iloc[:, 1:].values <= cor_p
    data = data.T
    data = np.repeat(data, 2, axis=1)
    data = np.repeat(data, 5, axis=0)
    cmap = plt.get_cmap('bwr')
    cmap.set_under('k', alpha=0)
    plt.imshow(data, cmap=cmap, vmin=0.5)
    plt.xticks(np.arange(len(x))*2, x, fontsize=2, rotation=90)
    plt.yticks(np.arange(len(y))*5+2, y, fontsize=4)
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

def acc_vs_P(dfs, cor_p, fpath, exp_name):
    labels = ['Pos', 'Neg', 'Rand']
    colors = ['blue', 'red', 'green']
    markers = ['o', '^', 'x']
    # labels = ['Na0', 'LayerNorm', 'Vanilla']
    # colors = ['blue', 'red', 'green']
    # markers = ['o', '^', 'x']
    xs = []
    data = []
    lines = []
    zs = []
    min_x, max_x = 10000, -10000
    for i, df in enumerate(dfs):
        xs.append(df['Acc'].values)
        min_x = min(min_x, np.min(xs[-1]))
        max_x = max(max_x, np.max(xs[-1]))
        data.append(-np.log10(df['P_Acc'].values))
        if i == 1:
            lines.append(plt.scatter(xs[-1], data[-1], label=labels[i], alpha=1, 
                s=3, marker=markers[i], c=colors[i]))
        else:
            lines.append(plt.scatter(xs[-1], data[-1], label=labels[i], alpha=0.5, 
                s=3, marker=markers[i], c=colors[i]))
        zs.append(np.polyfit(xs[-1], data[-1], 1))
    
    # x_ = np.linspace(min_x, max_x, 1000)
    # for i, z in enumerate(zs):
    #     plt.plot(x_, np.poly1d(z)(x_), ':', alpha=0.5, 
    #         c=lines[i].get_facecolors()[0])
    
    plt.gca().axhline(-np.log10(0.05), linestyle='--', alpha=0.5, 
        color='k', linewidth=0.5)
    plt.gca().axhline(-np.log10(cor_p), linestyle='--', alpha=0.5, 
        color='green', linewidth=0.5)
    # plt.gca().axvline(0.62, linestyle='--', alpha=0.5, 
    #     color='orange', linewidth=0.5)
    # plt.gca().axvline(0.5874, linestyle='--', alpha=0.5, 
    #     color='green', linewidth=0.5)
    # plt.gca().axvspan(0.5874, max(xs[0]+0.0005), 
    #     -np.log10(cor_p)/max(data[-1]) + 0.025, 1,
    #     alpha=0.2, facecolor='green', zorder=-1)
    
    plt.xlabel('Acc')
    plt.ylabel('-log10 (P)')
    plt.title('{} - Acc vs P'.format(exp_name))
    plt.legend()
    # fname = '{}_Acc_vs_P.svg'.format(fpath)
    fname = '{}_Acc_vs_P.png'.format(fpath)
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

def SNPs_vs_P(dfs, cor_p, fpath, exp_name):
    labels = ['Pos', 'Neg', 'Rand']
    colors = ['blue', 'red', 'green']
    markers = ['o', '^', 'x']

    xs = []
    data = []
    lines = []
    num_snps = dfs[0]['SNPs'].values
    xs = np.quantile(num_snps, np.arange(1, 11, 1)*0.1)
    xs = np.array(xs, dtype=int)
    print(xs)
    xs_ = []
    snps = []
    for i, df in enumerate(dfs):
        prev_x = 0
        for j, x in enumerate(xs):
            interval = x - prev_x
            expansion = 50/interval

            tdf = df.loc[(df['SNPs'] > prev_x) & (df['SNPs'] <= x)]
            data.append(-np.log10(tdf['P_Acc'].values))
            # xs_.extend(np.repeat(x, len(data[-1])))
            xs_.extend(tdf['SNPs'].values)
            # snps.extend((50*j)+(tdf['SNPs'].values - prev_x)*expansion)
            snps.extend(tdf['SNPs'].values)
            prev_x = x
            # print(round(min(snps)), round(max(snps)))
            # print()
        
        snps = np.array(snps) 
        # snps = -np.log(snps/max(snps))
        xs_ = snps
        data = np.concatenate(data)
        lines.append(plt.scatter(snps, data, color=colors[i], alpha=0.5,
                marker=markers[i], label=labels[i], s=4))
        z = np.polyfit(xs_, data, 1)
        x_ = np.linspace(min(xs_), max(xs_), 1000)
        # plt.plot(x_*(len(xs)*50/max(xs_)), np.poly1d(z)(x_), ':', alpha=0.5, 
        #     c=lines[i].get_facecolors()[0])
        plt.plot(x_, np.poly1d(z)(x_), ':', alpha=0.5, 
            c='r')
        snps = [] 
        data = []
        xs_ = []
    
    plt.gca().axhline(-np.log10(0.05), linestyle='--', alpha=0.3, 
        color='k', linewidth=0.5)
    plt.gca().axhline(-np.log10(cor_p), linestyle='--', alpha=0.3, 
        color='r', linewidth=0.5)
    # plt.xlabel('Num SNPs (ln)')
    plt.xlabel('Num SNPs')
    plt.ylabel('-log10 (P_Acc)')
    plt.title('{} - SNPs vs P'.format(exp_name))
    # print(xs)
    # plt.xticks(np.arange(50, (len(xs)+1)*50, 50), np.around(xs, 0), fontsize=4)
    # plt.xticks(np.arange(50, (len(xs)+1)*50, 50), np.around(xs, 0), fontsize=4)
    plt.legend()
    # fname = '{}_SNPs_vs_P.svg'.format(fpath)
    fname = '{}_SNPs_vs_P.png'.format(fpath)
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

def SNPs_vs_Acc(dfs, cor_p, fpath, exp_name):
    labels = ['Pos', 'Neg', 'Rand']
    colors = ['blue', 'red', 'green']
    markers = ['o', '^', 'x']
    # labels = ['Na0', 'LayerNorm', 'Vanilla']
    # colors = ['blue', 'red', 'green']
    # markers = ['o', '^', 'x']
    xs = []
    data = []
    lines = []
    num_snps = dfs[0]['SNPs'].values
    xs = np.array(xs, dtype=int)
    xs = np.quantile(num_snps, np.arange(1, 11, 1)*0.1)
    xs_ = []
    snps = []
    for i, df in enumerate(dfs):
        prev_x = 0
        for j, x in enumerate(xs):
            interval = x - prev_x
            expansion = 50/interval

            tdf = df.loc[(df['SNPs'] > prev_x) & (df['SNPs'] <= x)]
            data.append(tdf['Acc'].values)
            xs_.extend(np.repeat(x, len(data[-1])))
            # snps.extend((50*j)+(tdf['SNPs'].values - prev_x)*expansion)
            snps.extend(tdf['SNPs'].values)
            prev_x = x
            # print(round(min(snps)), round(max(snps)))
            # print()
        
        snps = np.array(snps) 
        # snps = np.log10(snps/max(snps))
        xs_ = snps

        data = np.concatenate(data)
        lines.append(plt.scatter(snps, data, color=colors[i], alpha=0.5,
                marker=markers[i], label=labels[i], s=4))
        z = np.polyfit(xs_, data, 1)
        x_ = np.linspace(min(xs_), max(xs_), 1000)
        # plt.plot(x_*(len(xs)*50/max(xs_)), np.poly1d(z)(x_), ':', alpha=0.5, 
        #     c=lines[i].get_facecolors()[0])
        plt.plot(x_, np.poly1d(z)(x_), ':', alpha=0.5, 
            c='r')
        snps = [] 
        data = []
        xs_ = []
     
    # plt.xlabel('Num SNPs (log10)')
    plt.xlabel('Num SNPs')
    plt.ylabel('Acc')
    plt.title('{} - SNPs vs Acc'.format(exp_name))
    # plt.xticks(np.arange(50, (len(xs)+1)*50, 50), np.around(xs, 0), fontsize=4)
    plt.legend()
    # fname = '{}_SNPs_vs_Acc.svg'.format(fpath)
    fname = '{}_SNPs_vs_Acc.png'.format(fpath)
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    
    # labels = ['Pos', 'Neg']
    # colors = ['blue', 'red']
    # markers = ['o', '^']
    # xs = []
    # data = []
    # lines = []
    # zs = []
    # min_x, max_x = 10000, 0
    # for i, df in enumerate(dfs):
    #     xs.append(np.log10(df['SNPs'].values))
    #     min_x = min(min_x, np.min(xs[-1]))
    #     max_x = max(max_x, np.max(xs[-1]))
    #     data.append(df['Acc'].values)
    #     lines.append(plt.scatter(xs[-1], data[-1], label=labels[i], alpha=0.8, 
    #         s=3, marker=markers[i], c=colors[i]))
    #     zs.append(np.polyfit(xs[-1], data[-1], 1))

    # x_ = np.linspace(min_x, max_x, 1000)
    # for i, z in enumerate(zs):
    #     plt.plot(x_, np.poly1d(z)(x_), ':', alpha=0.5, 
    #     c=lines[i].get_facecolors()[0])

    # plt.xlabel('log10 (Num_SNPs)')
    # plt.ylabel('Acc')
    # plt.title('{} - SNPs_vs_Acc.svg'.format(fpath))
    # plt.legend()
    # fname = '{}_SNPs_vs_Acc.svg'.format(fpath)
    # plt.savefig(fname, bbox_inches='tight')
    # plt.close()

def manhattan(dfs, cor_p, fpath, exp_name, genes_df):
    plt.figure(figsize=(10,7))

    xs = []
    data = []
    lines = []
    chrom_size_dict = {}
    prev = 0
    ticks = np.arange(1, 23)
    
    for c in range(1, 23):
        temp = [0,0] 
        temp[0] = prev
        temp[1] = prev + (np.max(genes_df.loc[genes_df['chrom'] == str(c)]['end']))/(10**7)
        prev = temp[1]
        chrom_size_dict[c] = temp
        ticks[c-1] = np.mean(temp)

    min_x = 0
    max_x = chrom_size_dict[22][1] + 1

    # colors = ['blue', 'green', 'red']
    # markers = ['o', '^', 'x']
    colors = np.tile(['mediumblue', 'deepskyblue'], 11)
    markers = np.repeat(['o'], 22)
    texts = []
    for i, df in enumerate(dfs):
        pos = genes_df.loc[df['Gene']]['start'].values
        pos = [chrom_size_dict[c][0] for c in df['Chrom'].values] + pos/(10**7)        
        x = pos
        xs.append(x)
        data.append(-np.log10(df['P_Acc'].values))
        lines.append(plt.scatter(xs[-1], data[-1], alpha=0.5, s=5, 
            marker=markers[i], color=colors[i]))
        for ind in range(len(xs[-1])):
            if df.iloc[ind]['Gene'] in ['PIGK', 'NRXN1', 'LRP1B', 'LYPD6B', 'PARD3B', 'RBMS3', 'STAC', 'FLNB', 'ROBO1', 'CADM2', 'EPHA6', 'CHCHD6', 'ATP1B3', 'SERPINI1', 'EFNA5', 'IMMP2L', 'EPHA1-AS1', 'CTNNA3', 'CNTN5', 'PITPNM2', 'GPR137C', 'TMEM170A', 'TOMM40', 'BCAM', 'APOC1', 'PPP1R37', 'EXOC3L2']:
                continue
            if data[-1][ind] >= -np.log10(cor_p):
                texts.append(plt.text(xs[-1][ind], data[-1][ind], df.iloc[ind]['Gene'], 
                    fontdict={'size':6}, rotation=90))
    adjust_text(texts, force_text=(0.4, 0.5), force_points=(0.5, 0.8),
                    arrowprops=dict(arrowstyle='-', color='k', lw=0.5))
    # plt.axhline(-np.log10(0.05), linestyles='--', alpha=0.5, colors='k', linewidth=0.5)
    plt.axhline(-np.log10(1e-5), linestyle='--', alpha=0.5, color='k', linewidth=0.5)
    plt.axhline(-np.log10(cor_p), linestyle='--', alpha=0.5, color='r', linewidth=0.5)
    plt.yticks(fontsize=14)
    # plt.ylim((0, max(np.concatenate(data))))
    plt.xticks(ticks, np.arange(1, 23), fontsize=14, rotation=90)
    plt.grid(axis='x', alpha=0.3)
    plt.xlabel('Chrom', fontsize=14)
    plt.ylabel('-log10 (P)', fontsize=14)
    plt.title('{} - manhattan'.format(exp_name))
    
    fname = '{}_manhattan.svg'.format(fpath)
    # fname = '{}_manhattan.png'.format(fpath)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

def acc_compare(dfs, cor_p, fpath, exp_name, exp_logs):
    for t, elogs, df in zip(['Pos', 'Neg'], exp_logs, dfs):
        print(t, elogs)
        fig, ax = plt.subplots(3, 1)
        ax = ax.flatten()
        data = [[],[],[]]
        gs = [[], [], []]
        for g in df['Gene'].to_list():
            print(g)
            if not os.path.isdir(os.path.join(elogs, g)):
                continue
            perm_a = np.load('{}/{}/ptest_metrics.npz'.format(elogs, g))['acc']
            p = df.loc[g]['P_Acc']
            # a = perm_a
            a = [perm_a[0],] + list(np.round(np.random.choice(perm_a, 999, replace=False), 2))
            if p <= cor_p:
                data[0].append(a)
                gs[0].append(g)
            elif p <= 0.05 and p > cor_p:
                data[1].append(a)
                gs[1].append(g)
            else:
                data[2].append(a)
                gs[2].append(g)
            
        for i in range(len(data)):
            print(i)
            d = np.asarray(data[i])
            print(d.shape)
            ax[i].violinplot(d.T)
            ax[i].scatter(np.arange(1, len(d)+1), d[:, 0], c='r', s=3, 
                marker='x')
            ax[i].set_xticks(np.arange(1, len(d)+1))
            ax[i].set_xticklabels(gs[i], fontsize=6, rotation=90)
            ax[i].grid(True, axis='x')
            ax[i].set_ylabel('Accuracy')
            if i == 0:
                ax[i].set_title('P <= {:3f}'.format(cor_p))
            elif i == 1:
                ax[i].set_title('P in [0.05, {:3f})'.format(cor_p))
            else:
                ax[i].set_title('P > 0.05')
        
        fig.tight_layout()
        fig.savefig('{}_{}_Acc_compare.svg'.format(fpath, t))
        plt.close()

def overfit_ratio(dfs, cor_p, fpath, exp_name, exp_logs):

    for t, elogs, df in zip(['Pos', 'Neg'], exp_logs, dfs):
        fig, ax = plt.subplots(4, 3, sharex=True)
        # ax = ax.flatten()
        data = []
        gs = [] 
        c = ['g', 'b', 'k', 'r']
        for g in df['Gene'].to_list():
            gp = os.path.join(elogs, g)
            if not os.path.isdir(gp):
                continue
            
            metrics = np.load('{}/{}/training_metrics.npz'.format(elogs, g))
            cm_train = metrics['agg_conf_mat'][0][:, 0]
            cm_val = metrics['agg_conf_mat'][0][:, 1]
            a_train = np.asarray([metrics_from_conf_mat(x)[3] for x in cm_train])
            a_val = np.asarray([metrics_from_conf_mat(x)[3] for x in cm_val])
            
            over_ratio = a_val/a_train
            mean_or = np.mean([np.mean(over_ratio[i*100:(i+1)*100]) for i in range(0,5)])
            mean_or = np.mean([np.mean(over_ratio[i*100:(i+1)*100]) for i in range(0,5)])
            
            if mean_or >= 1:
                i = 0
            elif mean_or < 1 and mean_or >= 0.9:
                i = 1
            elif mean_or < 0.9 and mean_or >= 0.8:
                i = 2
            else:
                i = 3        
            
            if df.loc[g]['P_Acc'] <= cor_p:
                ax[i, 0].plot(over_ratio, c=c[i], linewidth=0.5, alpha=0.5, 
                    linestyle='-')
                # ax[i, 1].text(len(over_ratio)-1, over_ratio[-1], '{}'.format(g), 
                #     fontdict=dict(fontsize=2))
            if df.loc[g]['P_Acc'] > cor_p and df.loc[g]['P_Acc'] <= 0.05:
                ax[i, 1].plot(over_ratio, c=c[i], linewidth=0.5, alpha=0.5, 
                    linestyle='-')
            else:
                ax[i, 2].plot(over_ratio, c=c[i], linewidth=0.5, alpha=0.3, 
                    linestyle=':')
            gs.append(g)

        ax[0, 0].set_title('P <= {:3f}'.format(cor_p))
        ax[0, 1].set_title('P in [0.05, {:3f})'.format(cor_p))
        ax[0, 2].set_title('P > 0.05')

        fig.tight_layout()
        fig.savefig('{}_{}_Overfit_ratio.svg'.format(fpath, t))
        plt.close()

def acc_overfit_ratio(dfs, cor_p, fpath, exp_name, exp_logs):
    for t, elogs, df in zip(['Pos', 'Neg'], exp_logs, dfs):
        data = [[],[],[]]

        data2 = [[],[],[]]
        y_off = [1, 1, 1]
        gs = [[], [], []] 
        c = ['g', 'b', 'k']
        for g in df['Gene'].to_list():
            gp = os.path.join(elogs, df.loc[g]['Gene_win'])
            if not os.path.isdir(gp):
                continue
            
            p = df.loc[g]['P_Acc']
            perm_a = np.load('{}/{}/ptest_metrics.npz'.format(elogs, df.loc[g]['Gene_win']))['acc']
            # a = perm_a
            a = [perm_a[0],] + list(np.round(np.random.choice(perm_a, 999, replace=False), 2))
            
            metrics = np.load('{}/{}/training_metrics.npz'.format(elogs, df.loc[g]['Gene_win']))
            cm_train = metrics['agg_conf_mat'][0][:, 0]
            cm_val = metrics['agg_conf_mat'][0][:, 1]
            a_train = np.asarray([metrics_from_conf_mat(x)[3] for x in cm_train])
            a_val = np.asarray([metrics_from_conf_mat(x)[3] for x in cm_val])
            
            o_r = a_val
            # /a_train
            if p <= cor_p:
                ind = 0
            elif p > cor_p and p <= 0.05:
                ind = 1
            else:
                ind = 2

            data[ind].append(a)
            gs[ind].append(g)
            o_r = (o_r - min(o_r))/(max(o_r) - min(o_r))
            o_r = (2*o_r) + (y_off[ind]-1)
            data2[ind].append(o_r)
            y_off[ind] += 2

        fig, ax = plt.subplots(3, 2, figsize=(10, 20), 
            gridspec_kw={'width_ratios': [1, 1], 
                        'height_ratios': [len(data[0])/len(df), 
                                len(data[1])/len(df), 
                                len(data[2])/len(df)]})

        for i in range(len(data)):
            # print(gs[i])
            d = np.asarray(data[i])
            d2 = np.asarray(data2[i])
            if len(d) == 0:
                continue
            [ax[i, 1].plot(d2_j, c=c[i], linewidth=0.5) for d2_j in d2]
            vp = ax[i, 0].violinplot(d.T, positions=np.arange(1, y_off[i], 2),
                vert=False, widths=2)
            vp['cmins'].set_linewidth(0)
            vp['cmaxes'].set_linewidth(0)
            vp['cbars'].set_linewidth(0.7)

            ax[i, 0].scatter(d[:, 0], np.arange(1, y_off[i], 2), c='r', s=5, 
                marker='x')
            ax[i, 0].scatter(1-d[:, 0], np.arange(1, y_off[i], 2), c='k', s=5, 
                marker='x', alpha=0.3)
            
            ax[i, 0].axvline(0.5, color='k', linestyle=':', linewidth=0.5, 
                alpha=0.6)
            ax[i, 0].set_xlim((0, 1))
            ax[i, 0].set_ylim((0, y_off[i]+1))
            ax[i, 0].set_yticks(np.arange(1, y_off[i], 2))
            ax[i, 0].set_yticklabels(gs[i], fontsize=4)
            ax[i, 0].grid(axis='y', color='r', linestyle=':', linewidth=0.5)
            
            ax[i, 1].set_ylim((0,y_off[i]+1))
            ax[i, 1].set_yticks(np.arange(1, y_off[i], 2))
            ax[i, 1].set_yticklabels([])
            ax[i, 1].grid(axis='y', color='r', linestyle=':', linewidth=0.5)

            if i == 0:
                ax[i, 0].set_ylabel('P <= {:3f}'.format(cor_p), fontsize=6)
            elif i == 1:
                ax[i, 0].set_ylabel('P in [0.05, {:3f})'.format(cor_p), fontsize=6)
            else:
                ax[i, 0].set_ylabel('P > 0.05', fontsize=8)

        ax[2, 0].set_xlabel('Accuracy')
        ax[2, 1].set_xlabel('Epochs')
        fig.tight_layout()
        fig.savefig('{}_{}_acc_overfit.svg'.format(fpath, t))
        plt.close()

def summary_plot(df, fname, totals=[]):
    if len(totals) == 0:
        fig, ax = plt.subplots(2, 2, sharex=True)
        ax = ax.flatten()
        if len(df.columns) > 5:
            totals = np.asarray([114, 114, 49, 49, 19, 19, 50, 50, 61, 61, 50, 50, 11, 11])
            confints = proportion_confint(df.iloc[:, 1:], totals, method='beta')
            df.iloc[:, 1:] = df.iloc[:, 1:]/totals
            df = df.iloc[:, 1:]

            x = np.arange(1, len(df)+1)*2
            ls = {'Unc':':', 'Corr':'-'}
            sym = {'Unc':'x', 'Corr':'o'}
            colours = {'Marioni':{'Neg':'red', 'Pos':'blue'},
                'KEGG':{'Neg':'orange', 'Pos':'green'},
                '11':{'Neg':'magenta'}, '19':{'Pos':'black'}}

            eps = {'Marioni':{'Neg':0.6, 'Pos':0},
                'KEGG':{'Neg':0.2, 'Pos':0.2},
                '11':{'Neg':0.6}, '19':{'Pos':0}}
            for i, c in enumerate(df.columns):
                try:
                    s, g, pn = c.split('_')
                except ValueError:
                    continue
                col = colours[g][pn]
                e = eps[g][pn]
                data = np.stack((confints[0][df.columns.to_list().index(c)], 
                    df[c].values, confints[1][df.columns.to_list().index(c)]), axis=-1)
                
                ind = 0
                if 'Marioni' in c or 'KEGG' in c:
                    if s == 'Unc':
                        ind = 0
                    else:
                        ind = 1
                else:
                    if s == 'Unc':
                        ind = 2
                    else:
                        ind = 3
                    
                ax[ind].boxplot(x=data.T, positions=x+e, showbox=False, widths=0,
                    medianprops={'color':col, 'marker':sym[s], 'markersize':2},
                    capprops={'color':col, 'marker':'_', 'markersize':2, 'alpha':0.3},
                    whiskerprops={'color':col, 'linewidth':0.5, 'alpha':0.3},
                    whis=0, showcaps=False, showfliers=False)
                # ax[1].scatter(x, df[c], label=c, c=col, marker=sym[s],
                    # s=9, alpha=0.5)
            ax[0].set_title('Marioni and KEGG (Unc)')
            ax[1].set_title('Marioni and KEGG (Corr)')
            ax[2].set_title('Old Group Train hits (Unc)')
            ax[3].set_title('Old Group Train hits (Corr)')
            # ax[0].legend(bbox_to_anchor=(1 ,1), loc='upper left', fontsize=7)
            # ax[1].legend(bbox_to_anchor=(1 ,1), loc='upper left', fontsize=7)
            for a in ax:
                a.axhline(0.05, linestyle=':', c='k')
                a.set_ylim(top=1)
                a.set_xticks([0,]+list(x))
                a.set_xticklabels(['',]+df.index.to_list(), dict(fontsize=6, rotation=90))
                
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()

        else:
            fig, ax = plt.subplots(1, 2)
            ax = ax.flatten()
            totals = np.asarray([120, 120, 120, 120])
            confints = proportion_confint(df.iloc[:, 1:], totals, method='beta')
            df.iloc[:, 1:] = df.iloc[:, 1:]/totals
            df = df.iloc[:, 1:]

            x = np.arange(1, len(df)+1)*2
            ls = {'Unc':':', 'Corr':'-'}
            sym = {'Unc':'x', 'Corr':'o'}
            colours = {'Neg':'red', 'Pos':'blue'}
    
            for i, c in enumerate(df.columns):
                try:
                    s, pn = c.split('_')
                except ValueError:
                    continue
                col = colours[pn]
                data = np.stack((confints[0][df.columns.to_list().index(c)], 
                    df[c].values, confints[1][df.columns.to_list().index(c)]), axis=-1)
                ind = 0
                if s == 'Unc':
                    ind = 0
                else:
                    ind = 1
                ax[ind].boxplot(x=data.T, positions=x, showbox=False, widths=0,
                    medianprops={'color':col, 'marker':sym[s], 'markersize':2},
                    capprops={'color':col, 'marker':'_', 'markersize':2, 'alpha':0.3},
                    whiskerprops={'color':col, 'linewidth':0.5, 'alpha':0.3},
                    whis=0, showcaps=False, showfliers=False)
            ax[0].set_title('Unc')
            ax[1].set_title('Corr')
            for a in ax:
                a.axhline(0.05, linestyle=':', c='k')
                a.set_ylim(top=1)
                a.set_xticks([0,]+list(x))
                a.set_xticklabels(['',]+df.index.to_list(), dict(fontsize=6, rotation=90))
                
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()

def binomial_plot(dfs, fname):
    
    dfs = sort_values
    
    
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax = ax.flatten()
    totals = np.asarray([114, 114, 49, 49, 19, 19, 50, 50, 61, 61, 50, 50, 11, 11])
    df.iloc[:, 1:] = 100*(df.iloc[:, 1:]/totals)
    df = df.iloc[:, 1:]

    x = np.arange(len(df))
    ls = {'Unc':':', 'Corr':'-'}
    sym = {'Unc':'x', 'Corr':'o'}
    colours = {'Marioni':{'Neg':'red', 'Pos':'blue'},
        'KEGG':{'Neg':'orange', 'Pos':'green'},
        '11':{'Neg':'magenta'}, '19':{'Pos':'black'}}
                
    for i, c in enumerate(df.columns):
        try:
            s, g, pn = c.split('_')
        except ValueError:
            continue
        col = colours[g][pn]

        confint = proportion_confint(df[c], 
            totals[df.columns.to_list().index(c)], method='beta')
        data = np.stack((confint[0], df[c].values, confint[1]), axis=-1)
        if 'Marioni' in c or 'KEGG' in c:
            # ax[0].boxplot(data, positions=x, showbox=False, widths=0,
            #     medianprops={'color':col, 'marker':sym[s], 's':9}, label=c)
            ax[0].scatter(x, df[c], label=c, c=col, marker=sym[s], s=9, alpha=0.5)
        else:
            # ax[1].boxplot(data, positions=x, showbox=False, widths=0,
            #     medianprops={'color':col, 'marker':sym[s], 's':9}, label=c)
            ax[1].scatter(x, df[c], label=c, c=col, marker=sym[s], s=9, alpha=0.5)

    ax[0].set_title('Marioni and KEGG')
    ax[1].set_title('Old Group Train hits')
    ax[0].legend(bbox_to_anchor=(1 ,1), loc='upper left', fontsize=7)
    ax[1].legend(bbox_to_anchor=(1 ,1), loc='upper left', fontsize=7)
    ax[0].axhline(0.05, linestyle=':', c='k')
    ax[1].axhline(0.05, linestyle=':', c='k')
    plt.xticks(x, df.index.values, fontsize=6, rotation=90)
    plt.savefig(fname)
    plt.close()
    #'beta' produces same results as R
    proportion_confint(method='beta') 

def gen_summary_documents(exp_base, exp_name, docs='1111'):
    with open('./params/system_specific_params.yaml', 'r') as params_file:
        sys_params = yaml.load(params_file, Loader=yaml.FullLoader)
    base_fold = sys_params['SUMMARY_BASE_FOLDER']
    exp_fold = '{}/{}'.format(base_fold, exp_name)
 
    genes_df = pd.read_csv('/home/upamanyu/GWASOnSteroids/GWASNN/datatables/genes.csv')
    genes_df.drop_duplicates(['symbol'], inplace=True)
    genes_df.set_index('symbol', drop=False, inplace=True)
    
    if not os.path.isdir(exp_fold):
        os.mkdir(exp_fold)

    cor_p = 3.397431541754434e-06

    # 1. Log Pos/Neg Corr+Uncorr ratios
    if int(docs[0]):
        summary_file = '{}/exp_summaries.csv'.format(base_fold)
        sd = {'Exp':exp_name}
        sd.update(ptest_pos_summary_stats(exp_base.format('Pos', 'Pos'), cor_p))
        sd.update(ptest_neg_summary_stats(exp_base.format('Neg', 'Neg'), cor_p))
        if os.path.isfile(summary_file):
            summary_df = pd.read_csv(summary_file)
            summary_df.set_index('Exp', drop=False, inplace=True)
            if exp_name not in summary_df.index.values:
                summary_df = summary_df.append(pd.DataFrame(sd, index=[exp_name,]))
                summary_df.to_csv(summary_file, index=False)
            else:
                print(exp_name, ' exists in summary file.')
        else:
            summary_df = pd.DataFrame(sd, index=[exp_name,])
            summary_df.to_csv(summary_file, index=False)
        summary_plot(summary_df, summary_file.replace('csv', 'svg'))
    
    # 2. Generate gradient plots (Diff plot)
    if int(docs[1]):
        
        for t in ['Pos', 'Neg']:
            fig_name = '{}/{}_{}_grads.svg'.format(exp_fold, t, exp_name)
            if os.path.isfile(fig_name):
                print(fig_name, ' exists')
                continue
            p_file = exp_base.format(t, t)
            model_fold = '/'.join(p_file.split('/')[:-1])
            df = pd.read_csv(p_file)
            
            df.sort_values('P_Acc', inplace=True)
            g_df = genes_df.loc[df.Gene.tolist()]
            g_df['SNPs'] = pd.Series()
            g_df.loc[df.Gene.tolist(), 'Perms'] = df['Perms'].values
            g_df.loc[df.Gene.tolist(), 'SNPs'] = df['SNPs'].values
            g_df.loc[df.Gene.tolist(), 'P_Acc'] = df['P_Acc'].values
            g_df.loc[df.Gene.tolist(), 'Type'] = df['Type'].values

            ds = []
            num_procs = 10
            d_size = (len(g_df)//num_procs)
            for i in range(0, num_procs):
                start = i*d_size
                end = (i+1)*d_size if i != num_procs-1 else len(g_df)
                d = {
                    'names': g_df.symbol.tolist()[start:end],
                    'chrom': g_df.chrom.tolist()[start:end],
                    'ids': g_df.id.tolist()[start:end]
                }
                ds.append([d, model_fold])
            gs, model_path, Xs, ys, snps, cws, colss = [], [], [], [], [], [], []
            with mp.Pool(num_procs) as pool:
                res = pool.starmap(return_data, ds)
                pool.close()
                for r in res:
                    gs.extend(r[0])
                    model_path.extend(r[1])
                    Xs.extend(r[2])
                    ys.extend(r[3])
                    snps.extend(r[4])
                    cws.extend(r[5])
                    colss.extend(r[6])
            num_covs = len([c for c in colss[0] if 'rs' not in c])
            gradient_plot(gs, model_path, Xs, ys, snps, cws, colss, num_covs,
                torch.device('cuda:3'), exp_fold, '{}_{}'.format(t, exp_name))
            
            # gradient_pair_plot(gs, g_df['P_Acc'].values, model_path, Xs, ys, 
            #     snps, cws, colss, num_covs, torch.device('cuda:3'), exp_fold, 
            #     '{}_{}'.format(t, exp_name))
            
            # Model structure graph
            if not os.path.isfile('{}/{}.png'.format(exp_fold, exp_name)):
                model = torch.load(model_path[0], map_location=torch.device('cpu'))
                raw_out = model.forward(torch.from_numpy(Xs[0]).float())
                make_dot(raw_out).render('{}/{}'.format(exp_fold, exp_name), 
                    format='png')
    
    # 3. Update hit_comparison plots (Diff plots for Neg and Pos)
    if int(docs[2]):
        
        for t in ['Pos', 'Neg']:
            p_file = exp_base.format(t, t)
            df = pd.read_csv(p_file)
            df.sort_values('Chrom', inplace=True)
            hit_comp_file = '{}/{}_hit_comparison.csv'.format(base_fold, t)
            if os.path.isfile(hit_comp_file):
                hits_df = pd.read_csv(hit_comp_file)
                hits_df.set_index('Gene', drop=False, inplace=True)
                hits_df.loc[df.Gene.values, exp_name] = df['P_Acc'].values
                hits_df.to_csv(hit_comp_file, index=False)
            else:
                hits_df = df[['Gene', 'P_Acc']]
                hits_df.rename(columns={'P_Acc':exp_name}, inplace=True)
                hits_df.to_csv(hit_comp_file, index=False)
            fig_name = hit_comp_file.replace('csv', 'svg')
            method_comparison(hits_df, cor_p, fig_name)

    # 4. Generate all summary plots (Single plot for Neg and Pos)
    if int(docs[3]):
        
        dfs = []
        exp_logs = []
        for t in ['Pos']:#, 'Neg', 'Rand']:
            p_file = exp_base.format(t, t)
            try:
                df = pd.read_csv(p_file)
            except FileNotFoundError:
                print('{} Does not exist'.format(p_file))
                continue
            df.set_index('Gene', drop=False, inplace=True)
            df = df.loc[~df['Gene'].isin(['APOE', 'TOMM40', 'APOC1'])]
            df['P_Acc'] = 10**(-1*df['P_Acc'].values)
            dfs.append(df)
            exp_logs.append('/'.join(p_file.split('/')[:-1]))
        fig_path = '{}/{}'.format(exp_fold, exp_name)

        dfs_c = [df.sort_values('Chrom') for df in dfs]
        # manhattan(dfs_c, cor_p, fig_path, exp_name, genes_df)
        SNPs_vs_P(dfs_c, cor_p, fig_path, exp_name)
        SNPs_vs_Acc(dfs_c, cor_p, fig_path, exp_name)
        acc_vs_P(dfs_c, cor_p, fig_path, exp_name)
        
        dfs_p = [df.sort_values('P_Acc') for df in dfs]
        # acc_compare(dfs_p, cor_p, fig_path, exp_name, exp_logs)
        # overfit_ratio(dfs_p, cor_p, fig_path, exp_name, exp_logs)
        # acc_overfit_ratio(dfs_p, cor_p, fig_path, exp_name, exp_logs)

# Functions to help understand the results better

def generate_GRS(gene_dict, summary_df, label, bp, beta_header='b'):

    maf_df = pd.read_csv('/mnt/sdb/Summary_stats/MAFs.csv')
    maf_df.drop_duplicates(subset=['SNP'], inplace=True)
    maf_df.set_index('SNP', inplace=True)
    
    test_df = pd.read_csv('./params/test_ids.csv', dtype={'iid':int})
    # train_df = pd.read_csv('./params/train_ids.csv', dtype={'iid':int})
    # test_df = pd.concat((test_df, train_df))
    
    test_df.drop_duplicates(['iid'], inplace=True)
    test_ids = test_df['iid'].values

    agg_gtypes = pd.DataFrame(columns=['iid', 'label'])
    agg_gtypes['iid'] = test_ids
    agg_gtypes.set_index('iid', inplace=True, drop=False)
    agg_gtypes.loc[test_ids, 'label'] = test_df[label].values

    snps = []
    beta_df = pd.DataFrame(columns=['SNP', beta_header])
    beta_df.set_index('SNP', drop=False, inplace=True)

    for i in range(len(gene_dict['names'])):
        g = gene_dict['names'][i]
        c = gene_dict['chrom'][i]

        data = pd.read_csv('{}/{}_chr{}_{}_{}bp.csv'.format(
            sys_params['DATA_BASE_FOLDER'], label, c, g, bp), 
            dtype={'iid':int})
        data.set_index('iid', inplace=True)
        
        # Find SNP with min P value in summary statistic
        gs = [s for s in data.columns if 'rs' in s]
        gs = [s.split('_')[1] for s in gs]
        gene_betas = summary_df.loc[summary_df.index.isin(gs)]
        min_p_beta = np.argmin(gene_betas['P'].values)
        g_snp = gene_betas.iloc[min_p_beta]['SNP']
        snps.extend([g_snp])
        beta_df.loc[g_snp, 'SNP'] = g_snp
        beta_df.loc[g_snp, beta_header] = float(gene_betas.iloc[min_p_beta][beta_header])
        
        data = data.loc[data.index.isin(test_ids)][str(c)+'_'+g_snp]
        agg_gtypes.loc[test_ids, g_snp] = data
        
        print(i, g, agg_gtypes.shape, len(snps))

    # Remove chromosome prefix from the snp columns
    agg_gtypes = agg_gtypes.loc[:, ~agg_gtypes.columns.duplicated()]
    
    # Get beta values for SNPs
    betas = beta_df.loc[snps][beta_header].values
    vprint("Summary data has {} missing SNPs.".format(len(snps)-len(betas)))
    snps = beta_df.index.values
    
    # Get the minor and major alleles from the bim file
    summary_alleles = summary_df.loc[snps][['A1', 'A2']].values
    bim_alleles = maf_df.loc[snps][['A1', 'A2']].values
    assert bim_alleles.shape == summary_alleles.shape
    beta_mask = np.where(bim_alleles == summary_alleles, 
        np.ones(bim_alleles.shape), np.ones(bim_alleles.shape)*-1)
    beta_mask = beta_mask[:, 0]
    print("Flipped alleles: {}".format(np.count_nonzero(beta_mask == -1)))
    betas = betas*beta_mask
    
    gtype_mat = agg_gtypes[snps].values
    gtype_mat[gtype_mat == 2] = -1 
    gtype_mat[gtype_mat == 0] = 2
    gtype_mat[gtype_mat == -1] = 0

    print("NANs: {}".format(np.count_nonzero(np.isnan(gtype_mat))))
    vprint("Betas shape: ", betas.shape)
    vprint("Gtype_mat shape: ", gtype_mat.shape)
    assert len(betas) == gtype_mat.shape[1]

    maf = maf_df.loc[snps]['MAF'].values
    assert len(maf) == gtype_mat.shape[1], print(len(maf))
    print("min MAF, max MAF: {} {}".format(np.min(maf), np.max(maf)))

    # Replace nan with 2*MAF for that SNP
    maf = np.tile(maf, len(gtype_mat)).reshape(gtype_mat.shape)
    maf = maf*2
    vprint("MAF shape: ", maf.shape)
    gtype_mat = np.where(np.isnan(gtype_mat), maf, gtype_mat)
    print("NANs: {}".format(np.count_nonzero(np.isnan(gtype_mat))))
    
    # Get GRS for each individual as Sum[B(snp)*Geno(SNP)]
    w_gtype_mat = np.multiply(gtype_mat, betas)
    grs = np.sum(w_gtype_mat, axis=1)
    grs = np.asarray([float(x) for x in grs])

    return grs, agg_gtypes['label'].values

def results_LD_plots(chrom, known_hits, nn_hits, label, bp):
    
    # Load hapmap LD file for the chromosome
    hapmap_ld_df = pd.read_csv('/mnt/sde/HapMap_LD/ld_chr{}_CEU.csv'.format(chrom))
    vprint('HapMap LD shape: {}'.format(hapmap_ld_df.shape))

    # Load bim file for the chromosome
    if chrom <= 10:
        bim_base = '/mnt/sdd/UKBB_1/'
    else:
        bim_base = '/mnt/sdc/UKBB_2/'
    bim_fname = '{}/ukb_imp_chr{}_v3_cleaned_geno_hwe_maf_2.bim'.format(
        bim_base, chrom)
    bim_df = pd.read_csv(bim_fname, sep='\t', header=None)
    bim_df.columns = ['chrom', 'snp', 'cm', 'pos', 'a1', 'a2']
    vprint('BIM shape: {}'.format(bim_df.shape))

    # Load gene df and keep only the genes of interest
    gene_df = pd.read_csv('./params/genes.csv')
    gene_df.set_index('symbol', drop=False, inplace=True)
    gene_df.sort_values(['start'], inplace=True)
    known_df = gene_df.loc[known_hits]
    nn_df = gene_df.loc[nn_hits]

    starts = known_df['start'].to_list() + nn_df['start'].to_list()
    ends = known_df['end'].to_list() + nn_df['end'].to_list()
    
    # Get all positions between the start and end of all genes
    pi = []
    [pi.extend(np.arange(i-50e3, j+50e3+1)) for i, j in zip(starts, ends)]
    pi = np.asarray(pi)
    vprint('Positin intervals shape: {}'.format(pi.shape))

    # Retain only the SNPs belonging to the genes
    ld_df = hapmap_ld_df.loc[hapmap_ld_df['pos1'].isin(pi)]
    ld_df = ld_df.loc[ld_df['pos2'].isin(pi)]
    ld_df = pd.concat((ld_df, ld_df.rename(columns={'pos1':'pos2', 'pos2':'pos1'})))
    ld_df = ld_df[['pos1', 'pos2', 'dprime', 'r2']]
    for p in np.unique(ld_df['pos1'].values):
        row = {'pos1':p, 'pos2':p, 'dprime':np.nan, 'r2':np.nan}
        ld_df = ld_df.append(row, ignore_index=True)
    
    bim_df = bim_df.loc[bim_df['pos'].isin(pi)]
    vprint('Filtered HapMap LD shape: {}'.format(ld_df.shape))
    vprint('Filtered BIM shape: {}'.format(bim_df.shape))
    
    ld_df.sort_values(['pos1', 'pos2'], inplace=True)
    # ld_mat = pd.pivot_table(ld_df, index='pos1', columns='pos2', values='r2')
    ld_mat = pd.pivot_table(ld_df, index='pos1', columns='pos2', values='dprime')
    vprint('dprime shape: {}'.format(ld_mat.shape))
    pos = ld_mat.index.values

    ax = plt.subplot(111)
    ax.imshow(ld_mat.values, cmap='Reds', origin='lower')
    
    ticks = []
    tick_labels = []
    tick_colors = []
    for g, s, e in zip(known_df['symbol'].values, known_df['start'].values, known_df['end'].values):
        si = bisect.bisect(pos, s)
        ei = bisect.bisect(pos, e)
        ax.axvspan(si, ei, color='g', alpha=0.2)
        tick_pos = si+(ei-si)//2
        if tick_pos == len(pos):
            tick_pos -= 1
        ticks.append(tick_pos)
        tick_labels.append('{:.3f}  {}'.format(pos[tick_pos]/1e6, g))
        tick_colors.append('g')

    for g, s, e in zip(nn_df['symbol'].values, nn_df['start'].values, nn_df['end'].values):
        si = bisect.bisect(pos, s)
        ei = bisect.bisect(pos, e)
        ax.axvspan(si, ei, color='orange', alpha=0.2)
        tick_pos = si+(ei-si)//2
        if tick_pos == len(pos):
            tick_pos -= 1
        ticks.append(tick_pos)
        tick_labels.append('{:.3f}  {}'.format(pos[tick_pos]/1e6, g))
        tick_colors.append('orange')
    
    sort_ind = np.argsort(ticks)
    ticks = [ticks[i] for i in sort_ind]
    tick_labels = [tick_labels[i] for i in sort_ind]
    tick_colors = [tick_colors[i] for i in sort_ind]

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=2, rotation=90)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=2)

    for ticklabel, tickcolor in zip(ax.get_xticklabels(), tick_colors):
        ticklabel.set_color(tickcolor)
    for ticklabel, tickcolor in zip(ax.get_yticklabels(), tick_colors):
        ticklabel.set_color(tickcolor)

    plt.savefig('../LD_Figures_AD/ld_mat_chrom{}.png'.format(chrom), dpi=600)
    plt.close()

