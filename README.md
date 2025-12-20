flowchart TD

    A[Raw CSV Data Sources] --> B[Staging & Cleaning Layer]
    
    B --> C1[Case 1: Promotion Effectiveness]
    B --> C2[Case 2: Loyalty Points Calculation]
    B --> C3[Case 3: Customer Segmentation - RFM]
    B --> C5[Case 5: Inventory Simulation]

    %% Case 1
    C1 --> C1a[Merge Sales + Promotions + Products]
    C1a --> C1b[Identify Promo using Transaction Date BETWEEN Start & End]
    C1b --> C1c[Calculate Promo vs Baseline Sales]
    C1c --> C1d[Sales Lift % & Revenue Lift %]
    C1d --> O1[Top 3 Effective Promotions]

    %% Case 2
    C2 --> C2a[Merge Sales + Customers]
    C2a --> C2b[Apply Tiered Points Rules]
    C2b --> C2c[Loyalty Status Multiplier]
    C2c --> C2d[Tenure Bonus]
    C2d --> C2e[Update Total Loyalty Points]

    %% Case 3
    C3 --> C3a[Aggregate Sales per Customer]
    C3a --> C3b[Calculate RFM Metrics]
    C3b --> C3c[High Spenders - Top 10%]
    C3b --> C3d[At-Risk Customers - High Recency]

    %% Case 4
    C2e --> C4[Case 4: Customer Notifications]
    C4 --> C4a[Identify Customers with New Points]
    C4a --> C4b[Generate Personalized Emails]
    C4b --> O2[Loyalty Notification Output]

    %% Case 5
    C5 --> C5a[Simulate Daily Store Inventory]
    C5a --> C5b[Identify Top 5 Best-Selling Products]

    %% Case 6
    C5b --> C6[Case 6: Inventory Impact Analysis]
    C6 --> C6a[Calculate Out-of-Stock Days]
    C6a --> C6b[Out-of-Stock Percentage]
    C6b --> C6c[Estimate Lost Revenue]
    C6c --> O3[Inventory Risk & Revenue Loss Report]
