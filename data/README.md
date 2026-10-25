# Used Car Valuation Dataset

## Dataset Attribution & Source
- **Primary Source**: Kaggle - "Vehicle Dataset from CarDekho" / "Car Details v3 Dataset"
- **Original Publisher**: CarDekho.com via Nehal Birla / Kaggle Community
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0) / Public Domain
- **Market Context**: Indian Used Automobile Market (Prices in Indian Rupees - ₹ / INR)

## Schema Description
The raw dataset contains the following attributes:

| Column Name | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `name` | String | Full car title (Brand + Model + Trim) | Maruti Swift Dzire VDI |
| `year` | Integer | Year of initial manufacturing / registration | 2014 |
| `selling_price` | Integer | Actual selling price in INR (₹) | 450000 |
| `km_driven` | Integer | Total cumulative distance driven in kilometers | 145500 |
| `fuel` | Categorical | Primary fuel type: Petrol, Diesel, CNG, LPG, Electric | Diesel |
| `seller_type` | Categorical | Channel of sale: Individual, Dealer, Trustmark Dealer | Individual |
| `transmission` | Categorical | Gearbox type: Manual, Automatic | Manual |
| `owner` | Categorical | Ownership tier: First Owner, Second Owner, etc. | First Owner |
| `mileage` | String/Float| Fuel efficiency (e.g. "23.4 kmpl", "18.5 km/kg") | 23.4 kmpl |
| `engine` | String/Float| Displacement volume in cubic centimetres (CC) | 1248 CC |
| `max_power` | String/Float| Peak engine output in brake horsepower (bhp) | 74 bhp |
| `torque` | String | Peak engine torque specification | 190Nm@ 2000rpm |
| `seats` | Float/Int | Total seating capacity | 5 |

## Cleaning & Feature Engineering
During preprocessing:
1. Strings with units (`kmpl`, `CC`, `bhp`) are extracted to clean numerical floating-point values.
2. The primary automobile manufacturer brand is parsed from the `name` column (e.g. "Maruti", "Hyundai", "Toyota").
3. Duplicate records are detected and logged.
4. Target leakage features (such as `torque` which has inconsistent text formats and minimal non-linear benefit) are pruned.
5. Numerical features undergo median imputation and standard scaling where appropriate.
6. Categorical features are one-hot encoded with handle_unknown='ignore' to guarantee zero inference crashing on unseen categories.
