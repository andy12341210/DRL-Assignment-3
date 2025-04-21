def reward_shaping(raw_reward, info, next_info, gamma=0.99):
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
    w_time     = 0.2   # 時間獎勵
    w_status   = 50.0    # 瑪麗狀態獎勵
    w_forward  = 1    # 前進獎勵
    w_death    = -30.0  # 死亡懲罰 (shaping 額外扣分)
    w_score    = 1.0

    status_value = {'small': 0, 'tall': 1, 'fireball': 2}
    # --- 定義 potential function Phi(s) ---
    def Phi(st):
        return (
            w_coin    * int(st['coins']) +
            w_flag    * int(st['flag_get']) +
            w_status  * status_value[st['status']] +
            w_time    * int(st['time']) +
            w_score   * int(st['score'])
        )

    # --- 1. potential-based shaping 項 ---
    F = gamma * Phi(next_info) - Phi(info)

    # --- 2. 直接 heuristic bonus: 金幣 & 前進 ---
    # coin_diff     = int(next_info['coins']) - int(info['coins'])
    # bonus_coin    = coin_diff * w_coin

    v             = int(next_info['x_pos']) - int(info['x_pos'])
    bonus_forward = v * w_forward * 0.5

    # if bonus_forward == 0:
    # v             = int(next_info['y_pos']) - int(info['y_pos'])
    # bonus_forward += max(v, 0) * w_forward * 0.7

    # --- 3. 死亡偵測 & 懲罰 ---
    life_diff     = int(next_info['life']) - int(info['life'])
    bonus_death   = w_death if life_diff < 0 else 0.0

    # --- 4. 組合所有項目 ---
    shaped = raw_reward + F  + bonus_forward + bonus_death

    return shaped