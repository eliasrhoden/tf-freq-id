import tf_freq_id


PT1 = tf_freq_id.PT1(1)
PT2 = tf_freq_id.PT2(10,1)
PT3 = tf_freq_id.PT2(22,1)

mdl = PT1/PT3 * PT1

print(mdl)



