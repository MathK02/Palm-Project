# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 20:28:24 2024

@author: Mathéo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from dataclasses import dataclass
import seaborn as sns

@dataclass
class PALMParameters:
    """Paramètres pour l'analyse PALM"""
    theta_0: float = 50.0    # Position réelle de l'émetteur
    delta_x: float = 2.0     # Pas de discrétisation
    sigma_b: float = 0.01    # Écart-type du bruit
    w: float = 2.0          # FWHM (Full Width at Half Maximum)
    N: int = 100            # Nombre de pixels
    a: float = 1.0          # Amplitude
    K: int = 1000           # Nombre d'itérations Monte Carlo
    
    @property
    def sigma_r(self):
        """Calcul de sigma_r à partir du FWHM"""
        return self.w / 2.355

class PALM1DAnalysis:
    def __init__(self, params: PALMParameters):
        self.params = params
        self.theta = np.linspace(0, 100, 10000)
        
    def r_i(self, theta, pixel_indices=None):
        """
        Calcule r_i(θ) pour un ou plusieurs pixels
        Args:
            theta: position du fluorophore
            pixel_indices: indices des pixels (si None, calcule pour tous les pixels)
        """
        if pixel_indices is None:
            pixel_indices = np.arange(self.params.N)
            
        x1 = (pixel_indices + 1) * self.params.delta_x - theta
        x0 = pixel_indices * self.params.delta_x - theta
        
        term1 = erf(x1 / (np.sqrt(2) * self.params.sigma_r))
        term2 = erf(x0 / (np.sqrt(2) * self.params.sigma_r))
        
        return 0.5 * (term1 - term2)

    def derivee_ri(self, theta, pixel_indices=None):
        """
        Calcule la dérivée de r_i par rapport à θ
        """
        if pixel_indices is None:
            pixel_indices = np.arange(self.params.N)
            
        x1 = (pixel_indices + 1) * self.params.delta_x - theta
        x0 = pixel_indices * self.params.delta_x - theta
        
        term1 = np.exp(-x1**2 / (2 * self.params.sigma_r**2))
        term2 = np.exp(-x0**2 / (2 * self.params.sigma_r**2))
        
        return -(term1 - term2) / (self.params.sigma_r * np.sqrt(2 * np.pi))

    def generate_signal(self, theta_0=None):
        """
        Génère un signal bruité pour une position θ donnée
        """
        if theta_0 is None:
            theta_0 = self.params.theta_0
            
        # Calcul du signal non bruité
        r = self.r_i(theta_0)
        # Ajout du bruit gaussien
        noise = np.random.normal(0, self.params.sigma_b, self.params.N)
        return self.params.a * r + noise

    def log_likelihood(self, signal, theta):
        """
        Calcule la log-vraisemblance pour un signal donné et une position θ
        """
        ri = self.r_i(theta)
        residuals = (signal - self.params.a * ri)**2
        return (-self.params.N * np.log(self.params.sigma_b) - 
                self.params.N * np.log(2 * np.pi)/2 - 
                np.sum(residuals)/(2 * self.params.sigma_b**2))

    def CRLB(self, theta):
        """
        Calcule la borne de Cramér-Rao pour une position θ
        """
        derivees = self.derivee_ri(theta)
        return self.params.sigma_b**2 / np.sum(derivees**2)

    def estimate_position(self, signal):
        """
        Estime la position par maximum de vraisemblance
        """
        log_l = [self.log_likelihood(signal, t) for t in self.theta]
        return self.theta[np.argmax(log_l)]

    def monte_carlo_analysis(self):
        """
        Réalise une analyse Monte Carlo complète
        """
        estimates = []
        for _ in range(self.params.K):
            # Génération du signal
            signal = self.generate_signal()
            # Estimation de la position
            theta_est = self.estimate_position(signal)
            estimates.append(theta_est)
        
        estimates = np.array(estimates)
        theoretical_crlb = np.sqrt(self.CRLB(self.params.theta_0))
        
        results = {
            'estimates': estimates,
            'mean': np.mean(estimates),
            'std': np.std(estimates),
            'bias': np.mean(estimates) - self.params.theta_0,
            'crlb': theoretical_crlb,
            'efficiency': theoretical_crlb / np.std(estimates)
        }
        
        return results

    def plot_complete_analysis(self):
        """
        Réalise et affiche une analyse complète
        """
        # Création de la figure avec plusieurs sous-plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Signal exemple
        signal = self.generate_signal()
        pixels = np.arange(self.params.N)
        ax1.plot(pixels, signal, 'b.', label='Signal bruité')
        ax1.plot(pixels, self.params.a * self.r_i(self.params.theta_0), 'r-', 
                label='Signal théorique')
        ax1.set_title('Exemple de signal')
        ax1.set_xlabel('Pixel')
        ax1.set_ylabel('Intensité')
        ax1.legend()
        ax1.grid(True)

        # 2. CRLB en fonction de la position
        crlb_values = [self.CRLB(t) for t in self.theta]
        ax2.plot(self.theta, crlb_values, 'g-')
        ax2.axvline(self.params.theta_0, color='r', linestyle='--', 
                   label='Position réelle')
        ax2.set_title('Borne de Cramér-Rao')
        ax2.set_xlabel('Position θ')
        ax2.set_ylabel('CRLB')
        ax2.legend()
        ax2.grid(True)

        # 3. Log-vraisemblance pour un signal
        log_l = [self.log_likelihood(signal, t) for t in self.theta]
        ax3.plot(self.theta, log_l, 'b-')
        ax3.axvline(self.params.theta_0, color='r', linestyle='--', 
                   label='Position réelle')
        ax3.set_title('Log-vraisemblance')
        ax3.set_xlabel('Position θ')
        ax3.set_ylabel('Log L(θ)')
        ax3.legend()
        ax3.grid(True)

        # 4. Analyse Monte Carlo avec compte simple
        results = self.monte_carlo_analysis()
        sns.histplot(data=results['estimates'], ax=ax4, stat='count', bins=30)
        ax4.set_title('Distribution des estimations Monte Carlo')
        ax4.set_xlabel('Position estimée')
        ax4.set_ylabel('Nombre d\'occurrences')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        
        # Affichage des résultats numériques
        print(f"""
Résultats de l'analyse Monte Carlo:
--------------------------------
Position réelle: {self.params.theta_0:.3f}
Position moyenne estimée: {results['mean']:.3f}
Écart-type empirique: {results['std']:.3f}
CRLB théorique: {results['crlb']:.3f}
Biais: {results['bias']:.3f}
Efficacité: {results['efficiency']:.3f}
""")
        
        return fig, results

# Exemple d'utilisation
if __name__ == "__main__":
    params = PALMParameters()
    analyzer = PALM1DAnalysis(params)
    fig, results = analyzer.plot_complete_analysis()
    plt.show()