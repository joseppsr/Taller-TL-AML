class FairLossPearson(keras.losses.Loss):
    """BCE + λ · corr(ŷ, s)²  (= CKA lineal, penaliza dependencia lineal)."""
    def __init__(self, lambda_fair=1.0, name='fair_pearson', **kwargs):
        super().__init__(name=name, **kwargs)
        self.lambda_fair = float(lambda_fair)

    def call(self, y_combined, y_pred):
        # y_combined empaqueta [y_true, s, sample_weight] para que el peso de clase
        # se aplique SOLO a la BCE y nunca reescale la penalización fair (lambda real fija).
        y_true = tf.cast(y_combined[:, 0:1], tf.float32)
        s      = tf.cast(y_combined[:, 1:2], tf.float32)
        w      = tf.cast(y_combined[:, 2:3], tf.float32)
        bce    = tf.keras.losses.binary_crossentropy(y_true, y_pred)  # (batch,)
        w_flat = tf.squeeze(w, axis=-1)
        weighted_bce = tf.reduce_sum(w_flat * bce) / (tf.reduce_sum(w_flat) + 1e-8)
        y_flat = tf.squeeze(y_pred, axis=-1)
        s_flat = tf.squeeze(s,      axis=-1)
        y_c = y_flat - tf.reduce_mean(y_flat)
        s_c = s_flat - tf.reduce_mean(s_flat)
        cov = tf.reduce_mean(y_c * s_c)
        std_y = tf.math.reduce_std(y_flat) + 1e-8
        std_s = tf.math.reduce_std(s_flat) + 1e-8
        corr  = cov / (std_y * std_s)
        # La penalización se estima sobre el mini-batch (estimador insesgado de corr²
        # con batch grande); devolvemos un escalar ya reducido (no usar sample_weight en fit).
        return weighted_bce + self.lambda_fair * tf.square(corr)

    def get_config(self):
        cfg = super().get_config()
        cfg['lambda_fair'] = self.lambda_fair
        return cfg


class FairLossCKARBF(keras.losses.Loss):
    """BCE + λ · CKA_RBF(ŷ, s)  (kernel gaussiano, captura dependencias no lineales).

    gamma=0.5 es una heurística razonable: ŷ∈[0,1] y s∈{0,1}, por lo que las distancias
    al cuadrado caen en [0,1] y un ancho de banda ~0.5 mantiene el kernel sensible en ese
    rango. Es un hiperparámetro que se podría calibrar (p.ej. heurística de la mediana)."""
    def __init__(self, lambda_fair=1.0, gamma=0.5, name='fair_cka_rbf', **kwargs):
        super().__init__(name=name, **kwargs)
        self.lambda_fair = float(lambda_fair)
        self.gamma       = float(gamma)

    def _cka_rbf(self, y, s):
        y_e = tf.expand_dims(y, 1)
        s_e = tf.expand_dims(s, 1)
        K = tf.exp(-tf.square(y_e - tf.transpose(y_e)) / (2.0 * self.gamma ** 2))
        L = tf.exp(-tf.square(s_e - tf.transpose(s_e)) / (2.0 * self.gamma ** 2))
        def center(M):
            return (M
                    - tf.reduce_mean(M, axis=1, keepdims=True)
                    - tf.reduce_mean(M, axis=0, keepdims=True)
                    + tf.reduce_mean(M))
        Kc, Lc = center(K), center(L)
        n  = tf.cast(tf.shape(y)[0], tf.float32)
        d2 = tf.square(n - 1.0)
        hsic_kl = tf.reduce_sum(Kc * Lc) / d2
        hsic_kk = tf.reduce_sum(Kc * Kc) / d2
        hsic_ll = tf.reduce_sum(Lc * Lc) / d2
        return tf.maximum(hsic_kl / (tf.sqrt(hsic_kk * hsic_ll) + 1e-8), 0.0)

    def call(self, y_combined, y_pred):
        # y_combined = [y_true, s, sample_weight]; el peso solo afecta a la BCE.
        y_true = tf.cast(y_combined[:, 0:1], tf.float32)
        s      = tf.cast(y_combined[:, 1:2], tf.float32)
        w      = tf.cast(y_combined[:, 2:3], tf.float32)
        bce    = tf.keras.losses.binary_crossentropy(y_true, y_pred)  # (batch,)
        w_flat = tf.squeeze(w, axis=-1)
        weighted_bce = tf.reduce_sum(w_flat * bce) / (tf.reduce_sum(w_flat) + 1e-8)
        # CKA_RBF estimado sobre el mini-batch; lambda_fair se mantiene constante.
        return weighted_bce + self.lambda_fair * self._cka_rbf(
            tf.squeeze(y_pred, axis=-1), tf.squeeze(s, axis=-1)
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'lambda_fair': self.lambda_fair, 'gamma': self.gamma})
        return cfg


print("Clases de pérdida definidas: FairLossPearson, FairLossCKARBF")
