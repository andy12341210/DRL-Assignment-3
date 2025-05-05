def reward_shaping(raw_reward, info, next_info, gamma=0.99, is_stuck=False):
    # if next_info['world'] > info['world']:
    #     raw_reward += 30
    return raw_reward
    """
    raw_reward: env.step 回傳的原始 reward (已 clip)
    info:       上一時刻的 info dict
    next_info:  執行 action 後的新狀態 info dict
    gamma:      discount，用於 potential-based shaping
    
    返回：shaped_reward
    """

    # --- 權重設定 (可自行調整) ---
    w_coin     = 20.0    # 金幣獎勵
    w_flag     = 1000.0   # 過關旗獎勵
    w_time     = 0.1   # 時間獎勵
    w_status   = 50.0    # 瑪麗狀態獎勵
    w_forward  = 1    # 前進獎勵
    w_death    = -20.0  # 死亡懲罰 (shaping 額外扣分)
    w_score    = 0.0

    # time_diff = next_info['time'] - info['time']
    # time_reduction = time_diff * w_time

    # score_diff     = int(next_info['score']) - int(info['score'])
    # bonus_score    = score_diff * w_score

    v             = float(next_info['x_pos']) - float(info['x_pos'])
    bonus_forward = v * w_forward
    # v             = float(next_info['y_pos']) - float(info['y_pos'])
    # bonus_forward += max(v, 0) * w_forward * 0.3

    # if is_stuck:
    #     bonus_forward = 0
    #     v             = float(next_info['y_pos']) - float(info['y_pos'])
    #     bonus_forward += max(v, 0) * w_forward * 0.5
    #     bonus_forward += (float(next_info['x_pos']) - float(info['x_pos'])) * w_forward * 1.5
    #     bonus_forward -= 8

    # --- 3. 死亡偵測 & 懲罰 --- 0.0
    life_diff     = int(next_info['life']) - int(info['life'])
    bonus_death   = w_death if life_diff < 0 else 0.0


    # --- 4. 組合所有項目 ---
    return raw_reward + bonus_forward + bonus_death

    # shaped = raw_reward + bonus_score + bonus_forward + bonus_death + time_reduction

    # return shaped,bonus_score,bonus_forward,bonus_death,time_reduction