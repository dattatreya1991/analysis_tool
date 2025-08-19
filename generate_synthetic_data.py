
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_data(start_date, end_date, dimensions, metrics, domain_type):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    data = []

    # Base values for metrics
    base_metrics = {
        'revenue': 1000,
        'spend': 500,
        'clicks': 10000,
        'impressions': 100000,
        'shoppers': 5000,
        'items_sold': 2000,
        'conversion_rate': 0.05,
        'bounce_rate': 0.30,
        'avg_order_value': 50
    }

    # Domain-specific adjustments
    if domain_type == 'ecommerce':
        dimension_values = {
            'business_unit': ['Electronics', 'Apparel', 'Home Goods'],
            'program': ['SEM', 'SEO', 'Affiliate', 'Email'],
            'geo': ['US', 'UK', 'CA', 'DE'],
            'device_type': ['Desktop', 'Mobile', 'Tablet'],
            'customer_segment': ['New', 'Returning', 'Loyal'],
            'product_category': ['Laptops', 'Smartphones', 'T-Shirts', 'Jeans', 'Furniture', 'Decor'],
            'entry_page': ['Homepage', 'Product Page', 'Category Page'],
            'campaign_ID': [f'CAMP_{i:03d}' for i in range(1, 11)]
        }
    elif domain_type == 'supply_chain':
        dimension_values = {
            'business_unit': ['Logistics', 'Warehousing', 'Procurement'],
            'program': ['Inbound', 'Outbound', 'Inventory Management'],
            'geo': ['North', 'South', 'East', 'West'],
            'device_type': ['Scanner', 'Tablet', 'Desktop'], # Simulating device types for operations
            'customer_segment': ['B2B', 'B2C'],
            'product_category': ['Raw Materials', 'Finished Goods', 'Components'],
            'entry_page': ['Warehouse Entry', 'Shipping Dock'], # Simulating entry points
            'campaign_ID': [f'SC_OPT_{i:03d}' for i in range(1, 11)] # Simulating optimization initiatives
        }
        # Adjust metrics for supply chain context
        base_metrics.update({
            'revenue': 50000, # e.g., revenue from logistics services
            'spend': 20000, # e.g., operational spend
            'clicks': 500, # e.g., system clicks/transactions
            'impressions': 5000, # e.g., data points processed
            'shoppers': 100, # e.g., unique shipments
            'items_sold': 10000, # e.g., units moved
            'conversion_rate': 0.95, # e.g., delivery success rate
            'bounce_rate': 0.05, # e.g., return rate
            'avg_order_value': 5000 # e.g., average shipment value
        })
    elif domain_type == 'fintech':
        dimension_values = {
            'business_unit': ['Retail Banking', 'Investment', 'Lending'],
            'program': ['Online Banking', 'Mobile App', 'Wealth Management'],
            'geo': ['Urban', 'Rural', 'Suburban'],
            'device_type': ['Mobile', 'Desktop', 'ATM'],
            'customer_segment': ['High Net Worth', 'Mass Market', 'SME'],
            'product_category': ['Savings', 'Checking', 'Loans', 'Investments'],
            'entry_page': ['Login', 'Account Summary', 'Application Form'],
            'campaign_ID': [f'FIN_PROMO_{i:03d}' for i in range(1, 11)]
        }
        # Adjust metrics for fintech context
        base_metrics.update({
            'revenue': 100000, # e.g., interest income, fees
            'spend': 30000, # e.g., operational costs
            'clicks': 50000, # e.g., app interactions
            'impressions': 500000, # e.g., ad views
            'shoppers': 10000, # e.g., unique users
            'items_sold': 5000, # e.g., transactions processed
            'conversion_rate': 0.10, # e.g., application conversion rate
            'bounce_rate': 0.20, # e.g., session bounce rate
            'avg_order_value': 1000 # e.g., average transaction value
        })
    else:
        raise ValueError("Invalid domain_type. Choose from 'ecommerce', 'supply_chain', 'fintech'.")

    # Filter dimensions based on user input
    selected_dimension_values = {dim: dimension_values[dim] for dim in dimensions}

    for single_date in date_range:
        # Seasonality (e.g., higher activity towards end of year)
        seasonality_factor = 1 + 0.2 * np.sin(single_date.dayofyear * 2 * np.pi / 365)

        # Campaign pauses (example: a dip in mid-year for SEM program)
        campaign_pause_factor = 1
        if domain_type == 'ecommerce' and 'program' in dimensions and 'SEM' in selected_dimension_values['program']:
            if single_date.month in [6, 7]: # June-July pause
                campaign_pause_factor = 0.7

        # Performance shifts (example: a permanent uplift after a certain date)
        performance_shift_factor = 1
        if single_date > datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=365): # After 1st year
            performance_shift_factor = 1.1 # 10% uplift

        # Generate data for each combination of selected dimensions
        from itertools import product
        for dim_combo_values in product(*selected_dimension_values.values()):
            row = {'Date': single_date}
            for i, dim_name in enumerate(dimensions):
                row[dim_name] = dim_combo_values[i]

            for metric in metrics:
                base_val = base_metrics.get(metric, 100) # Default if not in base_metrics
                noise = np.random.normal(0, base_val * 0.05) # 5% noise
                value = base_val * seasonality_factor * campaign_pause_factor * performance_shift_factor + noise

                # Ensure rates are within bounds
                if '_rate' in metric:
                    value = np.clip(value, 0.01, 0.99)
                elif metric == 'shoppers' or metric == 'items_sold':
                    value = max(1, round(value))
                else:
                    value = max(0, value)

                row[metric] = value
            data.append(row)

    return pd.DataFrame(data)

if __name__ == '__main__':
    start_date = '2023-01-01'
    end_date = '2024-12-31'

    # E-commerce data
    ecommerce_dimensions = ['business_unit', 'program', 'geo', 'device_type']
    ecommerce_metrics = ['revenue', 'spend', 'clicks', 'impressions', 'shoppers', 'conversion_rate']
    ecommerce_df = generate_data(start_date, end_date, ecommerce_dimensions, ecommerce_metrics, 'ecommerce')
    ecommerce_df.to_csv('ecommerce_data.csv', index=False)
    print("Generated ecommerce_data.csv")

    # Supply Chain data
    supply_chain_dimensions = ['business_unit', 'program', 'geo', 'product_category']
    supply_chain_metrics = ['revenue', 'spend', 'items_sold', 'conversion_rate', 'bounce_rate']
    supply_chain_df = generate_data(start_date, end_date, supply_chain_dimensions, supply_chain_metrics, 'supply_chain')
    supply_chain_df.to_csv('supply_chain_data.csv', index=False)
    print("Generated supply_chain_data.csv")

    # Fintech data
    fintech_dimensions = ['business_unit', 'program', 'customer_segment', 'product_category']
    fintech_metrics = ['revenue', 'spend', 'shoppers', 'avg_order_value', 'conversion_rate']
    fintech_df = generate_data(start_date, end_date, fintech_dimensions, fintech_metrics, 'fintech')
    fintech_df.to_csv('fintech_data.csv', index=False)
    print("Generated fintech_data.csv")




