# I am creating a training data set from scratch by plotting by hand.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


data_points = np.array([
    # [dam_level, flow_rate, power]
    # note these are rough estimates based on the graph
    # Dam level = 0
    [0, 0, 0],
    [0, 500, 0],
    [0, 1000, 0],
    [0, 1500, 10],
    [0, 2000, 10],
    [0, 2500, 40],
    [0, 3000, 45],
    [0, 4000, 45],

    # Dam level = 50
    [50, 0, 0],
    [50, 500, 0],
    [50, 1000, 0],
    [50, 1500, 10],
    [50, 2000, 10],
    [50, 2500, 40],
    [50, 3000, 45],
    [50, 4000, 45],

    # Dam level = 75
    [75, 0, 0],
    [75, 500, 0],
    [75, 1000, 0],
    [75, 1500, 10],
    [75, 2000, 10],
    [75, 2500, 40],
    [75, 3000, 50],
    [75, 4000, 50],

    # Dam level = 110
    [110, 0, 5],
    [110, 500, 5],
    [110, 1000, 30],
    [110, 1500, 50],
    [110, 2000, 55],
    [110, 2500, 60],
    [110, 3000, 90],
    [110, 4000, 100],

    # Dam level = 160
    [160, 0, 45],
    [160, 500, 45],
    [160, 1000, 55],
    [160, 1500, 75],
    [160, 2000, 120],
    [160, 2500, 110],
    [160, 3000, 110],
    [160, 4000, 120],

    # compating of 120 and 130 dam level as they differe mightily
    [120, 0, 5],
    [130, 0, 40],
    [120, 500, 5],
    [130, 500, 40],
    [120, 1000, 30],
    [130, 1000, 50],
    [120, 1500, 35],
    [130, 1500, 75],
    [120, 2000, 50],
    [130, 2000, 85],
    [120, 2500, 70],
    [130, 2500, 85],
    [120, 3000, 85],
    [130, 3000, 90],
    [120, 4000, 120],
    [130, 4000, 110],
])
# now i am just checking to make sure my data is all right based on my manuel inputs

# Extract columns
dam_level = data_points[:, 0]
flow_rate = data_points[:, 1]
power = data_points[:, 2]


print("Data ranges:")
print(f"Dam Level: {dam_level.min()} to {dam_level.max()}")
print(f"Flow Rate: {flow_rate.min()} to {flow_rate.max()}")
print(f"Power: {power.min()} to {power.max()}")
print(f"\nNumber of points: {len(data_points)}")

# Checking for obvious errors ... THIS SAVED ME BECAUSE I HAD AN ACCIDENTAL VALUE OF 500 instead of 50
if power.max() > 120 or power.min() < -10:
    print("WARNING: Power values seem off!")


# Now that I have this data, I am findign the equivelent of the Matlap Z slice so I can have 5 memberships, for 2 input, 1 output
# this is what I used: the guassian membership function,which is gradient-based


class GaussianMF:
    """Gaussian membership function"""

    def __init__(self, center, sigma):
        self.center = center
        self.sigma = sigma

    def compute(self, x):
        """Compute membership degree"""
        return np.exp(-0.5 * ((x - self.center) / self.sigma) ** 2)


class FuzzificationSystem:
    """
    Fuzzification system with 5 MFs per input 
    """

    def __init__(self):
        # Input 1: Dam Water Level (0-160)
        self.dam_mfs = self._create_mfs(n_mfs=5, input_range=(0, 160))

        # Input 2: Water Flow Rate (0-4000)
        self.flow_mfs = self._create_mfs(n_mfs=5, input_range=(0, 4000))

        # Linguistic labels for the 5 MFs
        self.labels = ["Very Low", "Low", "Medium", "High", "Very High"]

    def _create_mfs(self, n_mfs, input_range):
        """Create evenly-spaced Gaussian MFs"""
        x_min, x_max = input_range
        centers = np.linspace(x_min, x_max, n_mfs)
        sigma = (x_max - x_min) / (n_mfs - 1) / 2.0  # 50% overlap
        return [GaussianMF(c, sigma) for c in centers]
        # note, I think this is probably not the most accurate way, the overlap, probably needs to be optimized

    def fuzzify(self, dam_level, flow_rate):
        """
        Fuzzify the inputs

        Returns:
            dam_memberships: list of 5 membership degrees for dam level
            flow_memberships: list of 5 membership degrees for flow rate
        """
        dam_memberships = [mf.compute(dam_level) for mf in self.dam_mfs]
        flow_memberships = [mf.compute(flow_rate) for mf in self.flow_mfs]

        return dam_memberships, flow_memberships

    def visualize(self):
        """Visualize the membership functions"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot Dam Level MFs
        x1 = np.linspace(0, 160, 300)
        for i, mf in enumerate(self.dam_mfs):
            mu = [mf.compute(x) for x in x1]
            ax1.plot(x1, mu, label=self.labels[i], linewidth=2)

        ax1.set_xlabel('Dam Water Level', fontsize=12)
        ax1.set_ylabel('Membership Degree', fontsize=12)
        ax1.set_title(
            'Input 1: Dam Water Level Membership Functions', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([-0.05, 1.05])

        # Plot Flow Rate MFs
        x2 = np.linspace(0, 4000, 300)
        for i, mf in enumerate(self.flow_mfs):
            mu = [mf.compute(x) for x in x2]
            ax2.plot(x2, mu, label=self.labels[i], linewidth=2)

        ax2.set_xlabel('Water Flow Rate', fontsize=12)
        ax2.set_ylabel('Membership Degree', fontsize=12)
        ax2.set_title(
            'Input 2: Water Flow Rate Membership Functions', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([-0.05, 1.05])

        plt.tight_layout()
        return fig

    def demonstrate_fuzzification(self, dam_level, flow_rate):
        """
        Show how a specific input gets fuzzified
        """
        dam_mem, flow_mem = self.fuzzify(dam_level, flow_rate)

        print(f"\n{'='*60}")
        print(f"FUZZIFICATION EXAMPLE")
        print(f"{'='*60}")
        print(f"\nInput: Dam Level = {dam_level}, Flow Rate = {flow_rate}\n")

        print("Dam Water Level Memberships:")
        for i, (label, mem) in enumerate(zip(self.labels, dam_mem)):
            print(f"  MF{i+1} ({label:12s}): {mem:.4f}")

        print("\nWater Flow Rate Memberships:")
        for i, (label, mem) in enumerate(zip(self.labels, flow_mem)):
            print(f"  MF{i+1} ({label:12s}): {mem:.4f}")

        print(f"\n{'='*60}\n")


def main():
    """ Question 1"""

    print("===")
    print("QUESTION 1: FUZZIFICATION SYSTEM")
    print("===")

    # Create fuzzification system
    fuzz_system = FuzzificationSystem()

    # Demonstrate with a few example inputs
    print("\nDemonstrating fuzzification with example inputs:\n")

    # Example 1: Low dam, low flow
    fuzz_system.demonstrate_fuzzification(dam_level=30, flow_rate=500)

    # Example 2: Medium dam, high flow
    fuzz_system.demonstrate_fuzzification(dam_level=80, flow_rate=3000)

    # Example 3: High dam, medium flow
    fuzz_system.demonstrate_fuzzification(dam_level=130, flow_rate=2000)

    # Visualize the membership functions
    print("Creating membership function visualizations...")
    fig = fuzz_system.visualize()
    plt.savefig('question1_membership_functions.png',
                dpi=150, bbox_inches='tight')
    print("Saved: question1_membership_functions.png in current directory")

    # Print rationale
    print("\n" + "="*3)
    print("RATIONALE FOR MF SELECTION")
    print("="*3)
    print("""

    """)


if __name__ == "__main__":
    main()


# ============================================================================
# TSK SYSTEM TRAINING AND RULE BASE GENERATION
# ============================================================================

class TSKFuzzySystem:

    # this will learn consequent parameters via gradient descent

    def __init__(self, n_mfs_x1=5, n_mfs_x2=5):
        self.n_mfs_x1 = n_mfs_x1
        self.n_mfs_x2 = n_mfs_x2
        self.n_rules = n_mfs_x1 * n_mfs_x2

        # Use the membership functions from FuzzificationSystem
        fuzz_temp = FuzzificationSystem()
        self.mfs_x1 = fuzz_temp.dam_mfs
        self.mfs_x2 = fuzz_temp.flow_mfs

        # Initialize consequent parameters randomly
        np.random.seed(42)
        self.params = np.random.randn(self.n_rules, 3) * 0.1

        # Linguistic labels
        self.labels = ["Very Low", "Low", "Medium", "High", "Very High"]

    def fuzzify(self, x1, x2):
        """Compute firing strengths for all rules"""
        x1 = np.atleast_1d(x1)
        x2 = np.atleast_1d(x2)
        n_samples = len(x1)

        # Compute membership degrees
        mu_x1 = np.zeros((n_samples, self.n_mfs_x1))
        mu_x2 = np.zeros((n_samples, self.n_mfs_x2))

        for i, mf in enumerate(self.mfs_x1):
            mu_x1[:, i] = mf.compute(x1)

        for j, mf in enumerate(self.mfs_x2):
            mu_x2[:, j] = mf.compute(x2)

        # Compute firing strengths
        firing_strengths = np.zeros((n_samples, self.n_rules))
        rule_idx = 0
        for i in range(self.n_mfs_x1):
            for j in range(self.n_mfs_x2):
                firing_strengths[:, rule_idx] = mu_x1[:, i] * mu_x2[:, j]
                rule_idx += 1

        return firing_strengths

    def predict(self, x1, x2):
        """Predict output using TSK inference"""
        is_scalar = np.isscalar(x1)

        x1 = np.atleast_1d(x1)
        x2 = np.atleast_1d(x2)
        n_samples = len(x1)

        # calcuate firing strengths
        firing_strengths = self.fuzzify(x1, x2)

        #  outputs: y_i = p_i * x1 + q_i * x2 + r_i
        inputs = np.column_stack([x1, x2, np.ones(n_samples)])
        rule_outputs = inputs @ self.params.T

        # Weighted average defuzzification
        numerator = np.sum(firing_strengths * rule_outputs, axis=1)
        denominator = np.sum(firing_strengths, axis=1) + 1e-10

        y = numerator / denominator

        return float(y[0]) if is_scalar else y

    def compute_loss_and_gradients(self, x1, x2, y_true):
        """Compute MSE loss and gradients"""
        x1 = np.atleast_1d(x1)
        x2 = np.atleast_1d(x2)
        y_true = np.atleast_1d(y_true)
        n_samples = len(x1)

        # Forward pass
        firing_strengths = self.fuzzify(x1, x2)
        inputs = np.column_stack([x1, x2, np.ones(n_samples)])
        rule_outputs = inputs @ self.params.T

        numerator = np.sum(firing_strengths * rule_outputs, axis=1)
        denominator = np.sum(firing_strengths, axis=1) + 1e-10
        y_pred = numerator / denominator

        # MSE Loss
        loss = np.mean((y_pred - y_true) ** 2)

        # Gradient computation
        error = (y_pred - y_true)
        d_loss_d_ypred = 2 * error / n_samples
        d_ypred_d_rule_outputs = firing_strengths / denominator[:, np.newaxis]
        d_loss_d_rule_outputs = d_loss_d_ypred[:,
                                               np.newaxis] * d_ypred_d_rule_outputs

        gradients = np.zeros((self.n_rules, 3))
        for i in range(self.n_rules):
            gradients[i] = np.sum(
                d_loss_d_rule_outputs[:, i:i+1] * inputs, axis=0)

        return loss, gradients

    def train(self, x1_train, x2_train, y_train, n_epochs=2000, lr=0.001, verbose=True):
        """Train using gradient descent"""
        losses = []

        for epoch in range(n_epochs):
            loss, gradients = self.compute_loss_and_gradients(
                x1_train, x2_train, y_train)

            if np.isnan(loss) or np.any(np.isnan(gradients)):
                print(f"NaN detected at epoch {epoch+1}!")
                break

            # Clip gradients
            gradients = np.clip(gradients, -10, 10)

            # Update parameters
            self.params -= lr * gradients

            losses.append(loss)

            if verbose and (epoch + 1) % 500 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss:.4f}")

        return losses

    def print_learned_rules(self):
        """Print all 25 rules with learned coefficients"""
        print("\n" + "===")
        print("LEARNED TSK RULE BASE WITH COEFFICIENTS")
        print("===")
        print("\nAfter training with gradient descent, here are the learned rules:\n")

        rule_idx = 0
        for i in range(self.n_mfs_x1):
            for j in range(self.n_mfs_x2):
                p, q, r = self.params[rule_idx]
                dam_label = self.labels[i]
                flow_label = self.labels[j]

                print(f"Rule {rule_idx + 1}:")
                print(
                    f"  IF Dam Water Level is {dam_label:10s} AND Flow Rate is {flow_label:10s}")
                print(
                    f"  THEN Power = {p:.4f}×DamLevel + {q:.4f}×FlowRate + {r:.4f}")
                print()

                rule_idx += 1

        print("===")


# ===
# TRAINING of TSK SYSTEM
# ===

print("\n" + "===")
print("TRAINING TSK SYSTEM WITH GRADIENT DESCENT")
print("===")

# Create TSK system
tsk = TSKFuzzySystem(n_mfs_x1=5, n_mfs_x2=5)

print(f"\nSystem created with {tsk.n_rules} rules")
print(f"Parameters to learn: {tsk.n_rules * 3} (p, q, r for each rule)")

# Train the system
print("\nTraining...")
losses = tsk.train(dam_level, flow_rate, power,
                   n_epochs=2000, lr=0.001, verbose=True)

# Evaluate performance
y_pred = tsk.predict(dam_level, flow_rate)
mse = np.mean((y_pred - power) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_pred - power))

print(f"\nTraining Complete!")
print(f"Final MSE:  {mse:.4f}")
print(f"Final RMSE: {rmse:.4f}")
print(f"Final MAE:  {mae:.4f}")

# Print all learned rules
tsk.print_learned_rules()

# ============================================================================
# VISUALIZE LEARNED RULES IN A MATRIX
# ============================================================================


def create_learned_rule_matrix(tsk_system):
    """Create visual matrix with learned coefficients"""
    fig, ax = plt.subplots(figsize=(16, 12))

    labels = tsk_system.labels

    # Create grid
    rule_idx = 0
    for i in range(5):
        for j in range(5):
            x = j
            y = 4 - i

            # Rectangle for each rule
            rect = plt.Rectangle((x, y), 1, 1, linewidth=2,
                                 edgecolor='black', facecolor='lightblue', alpha=0.3)
            ax.add_patch(rect)

            # Get learned parameters
            p, q, r = tsk_system.params[rule_idx]

            # Add rule number
            ax.text(x + 0.5, y + 0.85, f'Rule {rule_idx + 1}',
                    ha='center', va='center', fontsize=12, fontweight='bold')

            # Add IF conditions
            ax.text(x + 0.5, y + 0.70, f'Dam: {labels[i]}',
                    ha='center', va='center', fontsize=8)
            ax.text(x + 0.5, y + 0.60, f'Flow: {labels[j]}',
                    ha='center', va='center', fontsize=8)

            # Add THEN equation
            ax.text(x + 0.5, y + 0.45, 'THEN:',
                    ha='center', va='center', fontsize=8, fontstyle='italic')
            ax.text(x + 0.5, y + 0.30, f'y = {p:.3f}×x₁',
                    ha='center', va='center', fontsize=7)
            ax.text(x + 0.5, y + 0.20, f'  + {q:.3f}×x₂',
                    ha='center', va='center', fontsize=7)
            ax.text(x + 0.5, y + 0.10, f'  + {r:.3f}',
                    ha='center', va='center', fontsize=7)

            rule_idx += 1

    # Axis settings
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.set_xlabel('Water Flow Rate →', fontsize=14, fontweight='bold')
    ax.set_ylabel('Dam Water Level →', fontsize=14, fontweight='bold')
    ax.set_title('Learned TSK Rule Base Matrix with Coefficients',
                 fontsize=16, fontweight='bold', pad=20)

    ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5])
    ax.set_yticklabels(labels, fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('learned_rule_base_matrix.png', dpi=150, bbox_inches='tight')
    print("\nSaved: learned_rule_base_matrix.png")


print("\nCreating visual rule matrix with learned coefficients...")
create_learned_rule_matrix(tsk)

# ============================================================================
# QUESTION 3: CONTROL SURFACE PLOT
# ============================================================================


print("\n" + "===")
print("QUESTION 3: GENERATING CONTROL SURFACE PLOT")
print("===")

# Create the control surface plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Create grid matching original data range
dam_grid = np.linspace(0, 160, 50)
flow_grid = np.linspace(0, 4000, 50)
Dam_mesh, Flow_mesh = np.meshgrid(dam_grid, flow_grid)

# Predict power output with TSK system
Power_mesh = np.zeros_like(Dam_mesh)
for i in range(Dam_mesh.shape[0]):
    for j in range(Dam_mesh.shape[1]):
        Power_mesh[i, j] = tsk.predict(Dam_mesh[i, j], Flow_mesh[i, j])

# Plot it
surf = ax.plot_surface(Dam_mesh, Flow_mesh, Power_mesh,
                       cmap='viridis', alpha=0.8,
                       linewidth=0.5, antialiased=True,
                       edgecolor='gray')

# Overlay training data points
ax.scatter(data_points[:, 0], data_points[:, 1], data_points[:, 2],
           c='red', marker='o', s=80, label='Training Data',
           edgecolors='black', linewidths=1.5, alpha=0.9)

# Labels and title
ax.set_xlabel('Dam Water Level', fontsize=14, labelpad=12)
ax.set_ylabel('Input Water Flow Rate', fontsize=14, labelpad=12)
ax.set_zlabel('Power Output', fontsize=14, labelpad=12)
ax.set_title('Control Surface: TSK Fuzzy System',
             fontsize=16, fontweight='bold', pad=25)

# Set same scale as original graph
ax.set_xlim(150, 0)  # Reversed to match original (150 at front, 0 at back)
ax.set_ylim(0, 4000)
ax.set_zlim(0, 140)

# colorbar
cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.1)
cbar.set_label('Power Output', fontsize=12)

# Set viewing angle to match original graph (view from front-right)
ax.view_init(elev=20, azim=-60)

# Add legend
ax.legend(loc='upper left', fontsize=11)

# Add grid
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('question3_control_surface.png', dpi=150, bbox_inches='tight')
print("Saved: question3_control_surface.png")

print("===")
print("QUESTION 3 COMPLETE")
print("===")
