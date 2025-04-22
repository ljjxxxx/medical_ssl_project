def get_disease_info(english_name):
    """
    返回每种疾病的描述和翻译

    Args:
        english_name: 疾病英文名

    Returns:
        info: 疾病信息字典
    """
    disease_dict = {
        'Atelectasis': {
            'name': 'Atelectasis',
            'translation': '肺不张',
            'description': '肺部或部分肺部的塌陷或闭合'
        },
        'Consolidation': {
            'name': 'Consolidation',
            'translation': '肺实变',
            'description': '肺组织被液体而非空气填充'
        },
        'Infiltration': {
            'name': 'Infiltration',
            'translation': '肺浸润',
            'description': '肺部存在不应出现的物质'
        },
        'Pneumothorax': {
            'name': 'Pneumothorax',
            'translation': '气胸',
            'description': '胸腔内有空气导致肺部塌陷'
        },
        'Edema': {
            'name': 'Edema',
            'translation': '肺水肿',
            'description': '肺组织中过多的液体'
        },
        'Emphysema': {
            'name': 'Emphysema',
            'translation': '肺气肿',
            'description': '肺部气泡（肺泡）受损'
        },
        'Fibrosis': {
            'name': 'Fibrosis',
            'translation': '肺纤维化',
            'description': '肺组织疤痕形成'
        },
        'Effusion': {
            'name': 'Effusion',
            'translation': '胸腔积液',
            'description': '肺部和胸腔之间的液体'
        },
        'Pneumonia': {
            'name': 'Pneumonia',
            'translation': '肺炎',
            'description': '肺部感染'
        },
        'Pleural_Thickening': {
            'name': 'Pleural Thickening',
            'translation': '胸膜增厚',
            'description': '胸膜（肺部周围的衬里）增厚'
        },
        'Cardiomegaly': {
            'name': 'Cardiomegaly',
            'translation': '心脏肥大',
            'description': '心脏扩大'
        },
        'Nodule': {
            'name': 'Nodule',
            'translation': '肺结节',
            'description': '肺部的小圆形生长物'
        },
        'Mass': {
            'name': 'Mass',
            'translation': '肺肿块',
            'description': '肺部较大的生长物或肿瘤'
        },
        'Hernia': {
            'name': 'Hernia',
            'translation': '疝气',
            'description': '器官通过腔壁的开口突出'
        }
    }

    return disease_dict.get(english_name, {'name': english_name, 'translation': '', 'description': ''})


def format_html_results(predictions):
    """
    格式化HTML结果显示

    Args:
        predictions: 疾病预测概率字典

    Returns:
        html: 格式化的HTML结果
    """
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

    html = "<div style='max-width: 800px; margin: 0 auto;'>"

    # 筛查模式说明
    html += f"""
    <div style='background-color: #f3f4f6; border-left: 4px solid #3b82f6; padding: 16px; margin-bottom: 20px; border-radius: 4px;'>
        <h3 style='margin-top: 0; color: #1d4ed8;'>筛查模式</h3>
        <p style='margin-bottom: 0;'>此模式优先考虑敏感性，使用较低的阈值(0.3)来减少漏诊风险。由于阈值降低，可能会出现一些假阳性结果，最终诊断需专业医生判断。</p>
    </div>
    """

    # 风险级别分类
    high_risk = []
    medium_high_risk = []
    medium_risk = []

    for disease, prob in sorted_preds:
        info = get_disease_info(disease)
        if prob >= 0.7:
            high_risk.append(f"{info['translation']} ({info['name']}): {prob:.3f}")
        elif prob >= 0.5:
            medium_high_risk.append(f"{info['translation']} ({info['name']}): {prob:.3f}")
        elif prob >= 0.3:
            medium_risk.append(f"{info['translation']} ({info['name']}): {prob:.3f}")

    # 高风险发现
    if high_risk:
        html += f"""
        <div style='background-color: #fef2f2; border-left: 4px solid #b91c1c; padding: 16px; margin-bottom: 20px; border-radius: 4px;'>
            <h3 style='margin-top: 0; color: #b91c1c;'>高风险发现 (≥0.7)</h3>
            <p style='margin-bottom: 0;'>{', '.join(high_risk)}</p>
            <p style='margin-top: 8px; font-size: 0.9em; color: #6b7280;'>强烈建议进一步专业检查</p>
        </div>
        """

    # 中高风险发现
    if medium_high_risk:
        html += f"""
        <div style='background-color: #fff5f5; border-left: 4px solid #ef4444; padding: 16px; margin-bottom: 20px; border-radius: 4px;'>
            <h3 style='margin-top: 0; color: #ef4444;'>中高风险发现 (0.5-0.7)</h3>
            <p style='margin-bottom: 0;'>{', '.join(medium_high_risk)}</p>
            <p style='margin-top: 8px; font-size: 0.9em; color: #6b7280;'>建议进一步专业检查</p>
        </div>
        """

    # 中度风险发现
    if medium_risk:
        html += f"""
        <div style='background-color: #fff7ed; border-left: 4px solid #f97316; padding: 16px; margin-bottom: 20px; border-radius: 4px;'>
            <h3 style='margin-top: 0; color: #f97316;'>中度风险发现 (0.3-0.5)</h3>
            <p style='margin-bottom: 0;'>{', '.join(medium_risk)}</p>
            <p style='margin-top: 8px; font-size: 0.9em; color: #6b7280;'>可能需要关注，建议咨询医生</p>
        </div>
        """

    # 无风险发现
    if not high_risk and not medium_high_risk and not medium_risk:
        html += f"""
        <div style='background-color: #f0fdf4; border-left: 4px solid #10b981; padding: 16px; margin-bottom: 20px; border-radius: 4px;'>
            <h3 style='margin-top: 0; color: #047857;'>未检测到显著风险</h3>
            <p style='margin-bottom: 0;'>未检测到超过筛查阈值(0.3)的显著发现。</p>
        </div>
        """

    html += """
    <h3>所有发现详情</h3>
    <table style='width: 100%; border-collapse: collapse;'>
        <thead>
            <tr style='background-color: #f3f4f6;'>
                <th style='text-align: left; padding: 8px; border-bottom: 2px solid #e5e7eb;'>发现</th>
                <th style='text-align: left; padding: 8px; border-bottom: 2px solid #e5e7eb;'>描述</th>
                <th style='text-align: center; padding: 8px; border-bottom: 2px solid #e5e7eb;'>概率</th>
                <th style='text-align: center; padding: 8px; border-bottom: 2px solid #e5e7eb;'>风险级别</th>
            </tr>
        </thead>
        <tbody>
    """

    for disease, prob in sorted_preds:
        info = get_disease_info(disease)

        # 确定风险级别和颜色
        if prob >= 0.7:
            risk_level = "高风险"
            status_color = "#b91c1c"
            bg_color = f"background-color: rgba(239, 68, 68, 0.15);"
        elif prob >= 0.5:
            risk_level = "中高风险"
            status_color = "#ef4444"
            bg_color = f"background-color: rgba(239, 68, 68, 0.1);"
        elif prob >= 0.3:
            risk_level = "中度风险"
            status_color = "#f97316"
            bg_color = f"background-color: rgba(249, 115, 22, 0.1);"
        else:
            risk_level = "低风险"
            status_color = "#10b981"
            bg_color = ""

        html += f"""
        <tr style='{bg_color} border-bottom: 1px solid #e5e7eb;'>
            <td style='padding: 8px;'><strong>{info['translation']}</strong><br><span style='color: #6b7280; font-size: 0.9em;'>{info['name']}</span></td>
            <td style='padding: 8px;'>{info['description']}</td>
            <td style='text-align: center; padding: 8px;'><span style='font-weight: {"bold" if prob >= 0.3 else "normal"};'>{prob:.3f}</span></td>
            <td style='text-align: center; padding: 8px;'>
                <span style='display: inline-block; padding: 4px 8px; border-radius: 9999px; background-color: {status_color}; color: white; font-size: 0.85em;'>
                    {risk_level}
                </span>
            </td>
        </tr>
        """

    html += """
        </tbody>
    </table>

    <div style='margin-top: 20px; padding: 12px; background-color: #f3f4f6; border-radius: 4px; font-size: 0.9em;'>
        <p style='margin: 0;'><strong>筛查模式说明：</strong>该系统使用较低阈值(0.3)来提高敏感性，减少漏诊风险。
        这可能导致一些假阳性结果。此AI分析仅供参考筛查，不构成医疗诊断。请务必咨询医疗专业人员以获取适当的诊断和治疗。</p>
    </div>
    </div>
    """

    return html