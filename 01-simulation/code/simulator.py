import pandas as pd
import numpy as np
import os
import uuid
import random
import faker
import tqdm

class UtilitySimulator:
    def __init__(self, output_dir='output', num_customers=1000, 
                 start_date = "01-01-2020", num_cycles=24, 
                 cycle_length=30, seed=None, description=None,
                 settings=None,
                 record_events_flag=True
    ):
        #  Initialize the simulator with parameters
        self.description = description if description else "Utility Billing Simulation"
        self.num_customers = num_customers
        self.start_date = pd.to_datetime(start_date)
        self.num_cycles = num_cycles
        self.cycle_length = cycle_length
        self.output_dir = output_dir
        self.record_events_flag = record_events_flag

        # Initialize data structures
        self.customers = []
        self.accounts = []
        self.usages = []
        self.billings = []
        self.payments = []
        self.balance_snapshots = []
        self.bad_debts = []
        self.events = []
        self.current_date = self.start_date
        self.accounts_realtime_data = {}

        if seed is not None:
            self.seed = seed
            random.seed(seed)
            np.random.seed(seed)
        if settings is None:
            self.settings = {
                'customer_reliability_distribution': {
                    'low': 0.2,
                    'medium': 0.5,
                    'high': 0.3
                },
                'usage_ranges': {
                    'low': (100, 300),
                    'medium': (200, 500),
                    'high': (400, 800)
                },
                'usage_probabilities': {
                    'low': 0.2,
                    'medium': 0.5,
                    'high': 0.3
                },
                'plan_rate': {
                    'residential': 0.22,
                    'commercial': 0.35,
                },
                'penalty_thresholds': {
                    'suspend': 5,
                    'close': 10,
                },
            }
        else:
            self.settings = settings

    def run_simulation(self):
        # Generate customers, accounts, usages, billings, payments, and events
        print("Starting simulation...")
        self.generate_customers()
        for cycle in tqdm.trange(self.num_cycles, position=0, desc="Simulation Cycles",  leave=False):
            for customer in tqdm.tqdm(self.customers, position=1, desc="Processing Customers", leave=False):
                self.simulate_account_creation(customer)

                for account in self.get_accounts_from_customer(customer['customer_id']):
                    # If account is just created created, nothing else to do but let it run through the first cycle
                    if self.current_date > account['start_date']:
                        # First, check 
                        self.check_account_delinquency_and_update_status(account)
                        self.generate_bill_for_usage_last_cycle(account)

                        if self.get_account_status(account['account_id']) in ['Active', 'Suspended']:
                            self.schedule_payment_for_latest_bill(account)
                            self.record_all_payments_this_cycle(account)

                    if self.get_account_status(account['account_id']) == 'Active':
                        self.simulate_account_usage_this_cycle(account)

            # Move to the next cycle
            self.current_date = self.get_next_cycle_start_date()

    def export_data(self, config_dict=None):
        # Export all data to CSV files
        self.customers_df = pd.DataFrame(self.customers).reset_index().rename(columns={"index": "customer_key"})
        self.accounts_df = pd.DataFrame(self.accounts).reset_index().rename(columns={"index": "account_key"})
        self.usages_df = pd.DataFrame(self.usages).reset_index().rename(columns={"index": "usage_key"})
        self.billings_df = pd.DataFrame(self.billings).reset_index().rename(columns={"index": "billing_key"})
        self.payments_df = pd.DataFrame(self.payments).reset_index().rename(columns={"index": "payment_key"})
        self.balance_snapshots_df = pd.DataFrame(self.balance_snapshots).reset_index().rename(columns={"index": "delinquency_key"})
        self.bad_debts_df = pd.DataFrame(self.bad_debts).reset_index().rename(columns={"index": "bad_debt_key"})

        self.customers_df["customer_key"] += 10**(int(np.log10(len(self.customers_df))) + 1) + 1
        self.accounts_df["account_key"] += 10**(int(np.log10(len(self.accounts_df))) + 1) + 1
        self.usages_df["usage_key"] += 1
        self.billings_df["billing_key"] += 1
        self.payments_df["payment_key"] += 1
        self.balance_snapshots_df["delinquency_key"] += 1
        self.bad_debts_df["bad_debt_key"] += 1

        # Hidden customer reliability from export
        customers_hidden_features = self.customers_df[['reliability']].copy()
        accounts_hidden_features = self.accounts_df[['usage_profile']].copy()

        self.customers_df = self.customers_df.drop(columns=['reliability'])
        self.accounts_df = self.accounts_df.drop(columns=['usage_profile'])

        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        # Export dataframes to CSV files
        self.customers_df.to_csv(os.path.join(self.output_dir, 'customers.csv'), index=False)
        self.accounts_df.to_csv(os.path.join(self.output_dir, 'accounts.csv'), index=False)
        self.usages_df.to_csv(os.path.join(self.output_dir, 'usages.csv'), index=False)
        self.billings_df.to_csv(os.path.join(self.output_dir, 'billings.csv'), index=False)
        self.payments_df.to_csv(os.path.join(self.output_dir, 'payments.csv'), index=False)
        self.balance_snapshots_df.to_csv(os.path.join(self.output_dir, 'balance_snapshots.csv'), index=False)
        self.bad_debts_df.to_csv(os.path.join(self.output_dir, 'bad_debts.csv'), index=False)
        customers_hidden_features.to_csv(os.path.join(self.output_dir, 'customers_hidden_features.csv'), index=False)
        accounts_hidden_features.to_csv(os.path.join(self.output_dir, 'accounts_hidden_features.csv'), index=False)
        print(f"Data exported to {self.output_dir}")

        if self.record_events_flag:
            self.events_df = pd.DataFrame(self.events).reset_index().rename(columns={"index": "event_key"}) 
            self.events_df["event_key"] += 1
            self.events_df.to_csv(os.path.join(self.output_dir, 'events.csv'), index=False)

        # Save config/args used for this simulation
        config_to_save = config_dict if config_dict is not None else {
            "num_customers": self.num_customers,
            "num_cycles": self.num_cycles,
            "cycle_length": self.cycle_length,
            "start_date": str(self.start_date),
            "output_dir": self.output_dir,
            "seed": getattr(self, "seed", None),
            "record_events_flag": self.record_events_flag,
            "settings": self.settings,
            "description": self.description
        }
        with open(os.path.join(self.output_dir, "simulator_config.yaml"), "w") as f:
            yaml.dump(config_to_save, f)


    def suspend_account(self, account):
        self.record_event(account, "suspend_account",None)

    def close_account(self, account):
        account_closure_record = {
            "account_id": account['account_id'],
            "account_status": self.get_account_status(account['account_id']),
            "closing_date": self.current_date,
            "closing_balance": self.get_account_balance(account['account_id']),
            "delinquency_score": self.get_account_delinquency_score(account['account_id']),
            "account_delinquency_status": self.get_account_delinquency_status(account['account_id']),
            "is_bad_debt": self.get_account_balance(account['account_id']) > 0
        }
        self.bad_debts.append(account_closure_record)
        self.record_event(account, "close_account", account_closure_record)


    def generate_customers(self):
        # Generate customer data
        for i in range(self.num_customers):
            customer_id = str(uuid.uuid4())
            customer_name = random_name()
            customer_type = "Residential" if random.random() < 0.8 else "Commercial"
            join_date = self.current_date
            reliability_profile = np.random.choice(
                ['low', 'medium', 'high'],
                p=[self.settings['customer_reliability_distribution']['low'],
                   self.settings['customer_reliability_distribution']['medium'],
                   self.settings['customer_reliability_distribution']['high']]
            )
            reliability = np.random.uniform(0.7, 1.0) if reliability_profile == 'high' else \
                np.random.uniform(0.4, 0.7) if reliability_profile == 'medium' else \
                np.random.uniform(0.1, 0.4)
            customer = {
                'customer_id': customer_id,
                'name': customer_name,
                'customer_type': customer_type,
                'join_date': join_date,
                'reliability': reliability,
            }
            self.customers.append(customer)

    def get_accounts_from_customer(self, customer_id):
        # Retrieve accounts for a given customer
        return [a for a in self.accounts if a["customer_id"] == customer_id]

    def simulate_account_creation(self, customer):
        # Simulate account creation for each customer
        accounts = self.get_accounts_from_customer(customer['customer_id'])

        # set probabilities for account creation
        # If no active accounts exist for the customer, make it very likely to create a new account
        if not accounts or all(self.get_account_status(a['account_id']) != 'Active' for a in accounts):
            account_creation_probability = 0.75
        elif len(accounts) <= 6:
            account_creation_probability = 0.05
        else:
            account_creation_probability = 0.02

        # Create 
        if random.random() < account_creation_probability:
            account_id = str(uuid.uuid4())
            usage_profile = np.random.choice(
                ['low', 'medium', 'high'],
                p=[self.settings['usage_probabilities']['low'],
                   self.settings['usage_probabilities']['medium'],
                   self.settings['usage_probabilities']['high']]
            )
            rate = self.settings['plan_rate'].get(customer['customer_type'].lower(), 0.22)
            account = {
                'account_id': account_id,
                'customer_id': customer['customer_id'],
                'usage_profile': usage_profile,
                'plan_rate': rate,
                'start_date': self.current_date
            }
            account_realtime_data = {
                'account_id': account_id,
                'balance': 0.0,
                'status': 'Active',
                'delinquency_status': 'None',
                "delinquency_score": 0,                
                'last_payment_date': None,
                'scheduled_payments': [],
                "interaction_order": 1,
            }
            self.accounts.append(account)
            self.accounts_realtime_data[account_id] = account_realtime_data
            self.record_event(account, "create_account", customer)

    def simulate_account_usage_this_cycle(self, account):
        # Simulate usage for the account
        usage = np.random.uniform(
            *self.settings['usage_ranges'][account['usage_profile']]
        )
        cycle_start = self.current_date
        cycle_end = self.current_date + pd.Timedelta(days=self.cycle_length - 1)
        usage_record = {
            'usage_id': str(uuid.uuid4()),
            'account_id': account['account_id'],
            'cycle_start': cycle_start,
            'cycle_end': cycle_end,
            'usage_kwh': usage
        }
        self.usages.append(usage_record)
        self.record_event(account, "simulate_usage", usage_record)

    def record_event(self, account, event_type, event_data=None):
        if self.record_events_flag is False:
            return
        if event_type == "create_account":
            customer = event_data
            event_record = {
                'event_id': str(uuid.uuid4()),
                'account_id': account['account_id'],
                'related_entity_id': customer['customer_id'],
                'event_type': 'Account Created',
                'timestamp': self.current_date,
                'details': f"Account created for {customer['name']}",
                'amount': 0.0,
                'balance': 0.0,
                'status': 'Active',
                'delinquency_score': 0,
                "interaction_order": 1,
            }
        elif event_type == "simulate_usage":
            usage_record = event_data
            usage = usage_record['usage_kwh']
            cycle_start = usage_record['cycle_start']
            cycle_end = usage_record['cycle_end']

            self.accounts_realtime_data[account['account_id']]['interaction_order'] += 1
            event_record = {
                'event_id': str(uuid.uuid4()),
                'account_id': account['account_id'],
                'related_entity_id': usage_record['usage_id'],
                'event_type': 'Usage Simulated',
                'timestamp': self.current_date,
                'details': f"Usage simulated: {usage:.2f} kWh for cycle {cycle_start.strftime('%Y-%m-%d')} - {cycle_end.strftime('%Y-%m-%d')}",
                'amount': 0,
                'balance': self.get_account_balance(account['account_id']),
                'status': self.get_account_status(account['account_id']),
                'delinquency_score': self.get_account_delinquency_score(account['account_id']),
                "interaction_order": self.get_account_interaction_order(account['account_id'])  
            }
        elif event_type == "check_delinquency":
            delinquency_check = event_data
            self.accounts_realtime_data[account['account_id']]['interaction_order'] += 1
            event_record = {
                'event_id': str(uuid.uuid4()),
                'account_id': account['account_id'],
                'related_entity_id': delinquency_check['delinquency_check_id'],
                'event_type': 'Delinquency Check',
                'timestamp': self.current_date,
                'details': f"Delinquency check performed: {delinquency_check['delinquency_status']} with score {delinquency_check['delinquency_score']} resulting in action {delinquency_check['account_action']}",
                'amount': 0,
                'balance': self.get_account_balance(account['account_id']),
                'status': self.get_account_status(account['account_id']),
                'delinquency_score': self.get_account_delinquency_score(account['account_id']),
                "interaction_order": self.get_account_interaction_order(account['account_id'])  
            }
        elif event_type == "generate_bill":
            billing_record = event_data
            self.accounts_realtime_data[account['account_id']]['interaction_order'] += 1
            event_record = {
                'event_id': str(uuid.uuid4()),
                'account_id': account['account_id'],
                'related_entity_id': billing_record['billing_id'],
                'event_type': 'Bill Generated',
                'timestamp': self.current_date,
                'details': f"Bill generated: {billing_record['new_charges']:.2f} for usage {billing_record['usage_kwh']:.2f} kWh from {billing_record['cycle_start'].strftime('%Y-%m-%d')} to {billing_record['cycle_end'].strftime('%Y-%m-%d')}",
                'amount': billing_record['new_charges'],
                'balance': self.get_account_balance(account['account_id']),
                'status': self.get_account_status(account['account_id']),
                'delinquency_score': self.get_account_delinquency_score(account['account_id']),
                "interaction_order": self.get_account_interaction_order(account['account_id'])  
            }
        elif event_type == "schedule_payment":
            scheduled_payment = event_data
            self.accounts_realtime_data[account['account_id']]['interaction_order'] += 1
            event_record = {
                'event_id': str(uuid.uuid4()),
                'account_id': account['account_id'],
                'related_entity_id': None,
                'event_type': 'Payment Scheduled',
                'timestamp': self.current_date,
                'details': f"Payment scheduled: {scheduled_payment['scheduled_payment_amount']:.2f} for bill {scheduled_payment['billing_id'][-12:]} on {scheduled_payment['scheduled_payment_date'].strftime('%Y-%m-%d')}",
                'amount': scheduled_payment['scheduled_payment_amount'],
                'balance': self.get_account_balance(account['account_id']),
                'status': self.get_account_status(account['account_id']),
                'delinquency_score': self.get_account_delinquency_score(account['account_id']),
                "interaction_order": self.get_account_interaction_order(account['account_id'])  
            }
        elif event_type == "record_payment":
            payment_record = event_data
            self.accounts_realtime_data[account['account_id']]['interaction_order'] += 1
            event_record = {
                'event_id': str(uuid.uuid4()),
                'account_id': account['account_id'],
                'related_entity_id': payment_record['payment_id'],
                'event_type': 'Payment Recorded',
                'timestamp': self.current_date,
                'details': f"Payment recorded: {payment_record['payment_amount']:.2f} on {payment_record['payment_date'].strftime('%Y-%m-%d')} with new balance {payment_record['new_balance']:.2f}",
                'amount': payment_record['payment_amount'],
                'balance': self.get_account_balance(account['account_id']),
                'status': self.get_account_status(account['account_id']),
                'delinquency_score': self.get_account_delinquency_score(account['account_id']),
                "interaction_order": self.get_account_interaction_order(account['account_id'])  
            }
        elif event_type == "suspend_account":
            self.accounts_realtime_data[account['account_id']]['interaction_order'] += 1
            event_record = {
                'event_id': str(uuid.uuid4()),
                'account_id': account['account_id'],
                'related_entity_id': None,
                'event_type': 'Account Suspended',
                'timestamp': self.current_date,
                'details': f"Account {account['account_id']} suspended with balance {self.get_account_balance(account['account_id']):.2f} & delinquency score {self.get_account_delinquency_score(account['account_id'])}",
                'amount': 0.0,
                'balance': self.get_account_balance(account['account_id']),
                'status': self.get_account_status(account['account_id']),
                'delinquency_score': self.get_account_delinquency_score(account['account_id']),
                "interaction_order": self.get_account_interaction_order(account['account_id'])  
            }
        elif event_type == "close_account":
            account_closure_record = event_data
            self.accounts_realtime_data[account['account_id']]['interaction_order'] += 1
            event_record = {
                'event_id': str(uuid.uuid4()),
                'account_id': account['account_id'],
                'related_entity_id': None,
                'event_type': 'Account Closed',
                'timestamp': self.current_date,
                'details': f"Account {account['account_id']} closed with balance {account_closure_record['closing_balance']:.2f} & delinquency score {account_closure_record['delinquency_score']} bad debt status {account_closure_record['is_bad_debt']}",
                'amount': 0.0,
                'balance': self.get_account_balance(account['account_id']),
                'status': self.get_account_status(account['account_id']),
                'delinquency_score': self.get_account_delinquency_score(account['account_id']),
                "interaction_order": self.get_account_interaction_order(account['account_id'])  
            }

        self.events.append(event_record)


    def check_account_delinquency_and_update_status(self, account):
        # Check if the account is delinquent
        # Get the last bill for the account
        if self.get_account_status(account['account_id']) == 'Closed':
            # If the account is closed, no need to check delinquency
            return
        last_bill = self.get_last_bill_for_account(account['account_id'])
        last_delinquency_check = self.get_last_delinquency_check_for_account(account['account_id'])
        if not last_bill:
            # No bill available, cannot check delinquency
            return
        
        # Information from the last bill - amount and date due
        billing_id = last_bill['billing_id']
        total_balance = last_bill['total_balance']
        prev_unpaid_balance = last_bill.get("carryover_balance", 0.0)
        date_due = last_bill['date_due']

        if self.current_date < date_due:
            # If the current date is before the latest bill due date, no delinquency check needed
            return

        # Information from the last delinquency check
        if last_delinquency_check:
            prev_delinquency_score = last_delinquency_check['delinquency_score']
        else:
            prev_delinquency_score = 0

        # Get the current balance and delinquency score
        current_balance = self.get_account_balance(account['account_id'])
        current_delinquency_score = self.get_account_delinquency_score(account['account_id'])
        current_status = self.get_account_status(account['account_id'])
        current_delinquency_status = self.get_account_delinquency_status(account['account_id'])

        # Check if the account is delinquent
        unpaid_balance_ratio = current_balance / total_balance if total_balance > 0 else 0
        account_action = 'No Action'
 
        if current_balance <= 0:
            new_account_status = 'Active'
            new_delinquency_status = 'None'
            new_delinquency_score = 0
            is_delinquent = False
            delinquency_penalty = 0
            delinquency_type = "None"
            if prev_unpaid_balance > 0:
                # If the account was previously delinquent but now has a zero balance, reset the delinquency status
                if current_delinquency_status == 'Delinquent':
                    account_action = 'Cleared Delinquency'
                elif current_delinquency_status == 'Suspended':
                    account_action = 'Cleared Suspension'
                elif current_delinquency_status == 'Closed':
                    account_action = 'Cleared Closure'
                else:
                    account_action = 'No Action'

        else:
            is_delinquent = True
            if unpaid_balance_ratio == 1:
                delinquency_penalty = 2
                delinquency_type = "Full"
            elif unpaid_balance_ratio >= 0.67:
                delinquency_penalty = 1.5
                delinquency_type = "Major"
            elif unpaid_balance_ratio >= 0.33:
                delinquency_penalty = 1
                delinquency_type = "Partial"
            elif unpaid_balance_ratio > 0:
                delinquency_penalty = 0.5
                delinquency_type = "Minor"
            else:
                delinquency_penalty = 0
                delinquency_type = "None"
            new_delinquency_score = current_delinquency_score + delinquency_penalty

            if new_delinquency_score >= self.settings['penalty_thresholds']['close']:
                new_account_status = 'Closed'
                new_delinquency_status = 'Closed'
                account_action = 'Closed Account'
            elif new_delinquency_score >= self.settings['penalty_thresholds']['suspend']:
                new_account_status = 'Suspended'
                new_delinquency_status = 'Suspended'
                account_action = 'Suspended Account'
            elif new_delinquency_score > 0:                    
                new_account_status = 'Active'
                new_delinquency_status = 'Delinquent'
                account_action = 'Account Marked Delinquent' if prev_delinquency_score <= 0 else 'Account Continued Delinquency'
            else:
                new_account_status = 'Active'
                new_delinquency_status = 'None'
                account_action = 'No Action'

        # Update acccount status 
        self.set_account_status(account['account_id'], new_account_status)
        self.set_account_delinquency_status(account['account_id'], new_delinquency_status)
        self.set_account_delinquency_score(account['account_id'], new_delinquency_score)

        delinquency_check_record = {
            'delinquency_check_id': str(uuid.uuid4()),
            'billing_id': billing_id,
            'account_id': account['account_id'],
            'account_status': new_account_status,
            'check_date': self.current_date,
            'total_due': total_balance,
            'date_due': date_due,
            'unpaid_balance': current_balance,
            'unpaid_balance_ratio': unpaid_balance_ratio,
            'prev_unpaid_balance': prev_unpaid_balance,
            'prev_delinquency_score': prev_delinquency_score,
            'is_delinquent': is_delinquent,
            'delinquency_type': delinquency_type,
            'delinquency_penalty': delinquency_penalty,
            'delinquency_score': new_delinquency_score,
            'delinquency_status': new_delinquency_status,
            'account_action': account_action,
        }
        self.balance_snapshots.append(delinquency_check_record)
        self.record_event(account, "check_delinquency", delinquency_check_record)

        if account_action == 'Closed Account':
            self.close_account(account)
        elif account_action == 'Suspended Account':
            self.suspend_account(account)


    def generate_bill_for_usage_last_cycle(self, account):
        last_usage = self.get_last_usage_for_account(account['account_id'])
        last_bill = self.get_last_bill_for_account(account['account_id'])
        payments_last_cycle = self.get_payments_last_cycle_for_account(account['account_id'])
        
        if not last_usage:
            # No usage data available for the last cycle
            return
        
        # Verify that usage is for last cycle
        # If the bill cycle end is within the last cycle (aka 30 days), then we can generate a bill
        if last_usage['cycle_end'] >= (self.current_date - pd.Timedelta(days=self.cycle_length)):
            # Generate a bill for the last cycle's usage
            usage_kwh = last_usage['usage_kwh']
        else:
            # No usage data available for the last cycle, either since the account was just created or suspended/closed
            usage_kwh = 0

        cycle_start = last_usage['cycle_start'] if last_usage else self.current_date - pd.Timedelta(days=self.cycle_length)
        cycle_end = last_usage['cycle_end'] if last_usage else self.current_date - pd.Timedelta(days=1)
        plan_rate = account['plan_rate']
        new_charges = usage_kwh * plan_rate
        previous_balance = last_bill['total_balance'] if last_bill else 0
        payment_total = sum(p['payment_amount'] for p in payments_last_cycle) if payments_last_cycle else 0
        carryover = round(previous_balance - payment_total, 2)
        total_balance = round(carryover + new_charges, 2)

        # Update the account's realtime data
        self.set_account_balance(account['account_id'], total_balance)

        billing_record = {
            "billing_id": str(uuid.uuid4()),
            "account_id": account['account_id'],
            "cycle_start": cycle_start,
            "cycle_end": cycle_end,
            "usage_kwh": usage_kwh,
            "plan_rate": plan_rate,
            "previous_balance": previous_balance,
            "payment_total": payment_total,
            "carryover_balance": carryover,
            "new_charges": new_charges,
            "total_balance": total_balance,
            "date_issued": self.current_date,
            "date_due": self.get_next_cycle_start_date(),
        }
        self.billings.append(billing_record)
        self.record_event(account, "generate_bill", billing_record)

    def schedule_payment_for_latest_bill(self, account):
        # Schedule a payment for the latest bill
        last_bill = self.get_last_bill_for_account(account['account_id'])
        customer = self.get_customer_by_id(account['customer_id'])
        customer_reliability = customer['reliability'] if customer else 0.5  # Default reliability if customer not found
        
        scheduled_payments = self.get_scheduled_payments_for_account(account['account_id'])
        scheduled_payments_this_cycle = [p for p in scheduled_payments if self.current_date <= p['scheduled_payment_date'] < self.get_next_cycle_start_date()]
        total_amount_scheduled = sum(p['scheduled_payment_amount'] for p in scheduled_payments_this_cycle) if scheduled_payments_this_cycle else 0

        if not last_bill:
            return
        
        date_due = last_bill['date_due']
        total_balance = last_bill['total_balance']
        remaining = total_balance - total_amount_scheduled

        if remaining > 0:
            # Simulate payment amount and date based on customer reliability
            max_days_late = int((1.2 - customer_reliability) * 100)
            days_late = int(random.gauss(mu=10 - 50 * customer_reliability, sigma=15))
            days_late = min(max(days_late, -15), max_days_late)
            payment_date = date_due + pd.Timedelta(days=days_late)

            if days_late > 60 and customer_reliability < 0.75:
                return None

            # Partial/full decision
            if days_late <= 7:
                payment_amount = remaining if random.random() < 0.9 else round(remaining * random.uniform(0.6, 0.9), 2)
            else:
                if customer_reliability > 0.85:
                    payment_amount = round(remaining * random.uniform(0.6, 1.0), 2)
                elif customer_reliability > 0.7:
                    payment_amount = round(remaining * random.uniform(0.3, 0.6), 2)
                else:
                    payment_amount = round(remaining * random.uniform(0.1, 0.3), 2)

            new_scheduled_payment = {
                'billing_id': last_bill['billing_id'],
                'scheduled_payment_amount': payment_amount,
                'scheduled_payment_date': payment_date,
            }
            self.add_scheduled_payment_for_account(account['account_id'], new_scheduled_payment)
            self.record_event(account, "schedule_payment", new_scheduled_payment)            

    def record_all_payments_this_cycle(self, account):
        scheduled_payments = self.get_scheduled_payments_for_account(account['account_id'])
        scheduled_payments_this_cycle = [p for p in scheduled_payments if self.current_date <= p['scheduled_payment_date'] < self.get_next_cycle_start_date()]
        scheduled_payments_this_cycle.sort(key=lambda x: x['scheduled_payment_date'])         # Sort scheduled payments by date

        for scheduled_payment in scheduled_payments_this_cycle:
            # Record the account balance before payment
            payment_date = scheduled_payment['scheduled_payment_date']
            payment_amount = scheduled_payment['scheduled_payment_amount']
            prev_account_balance = self.get_account_balance(account['account_id'])
            new_account_balance = prev_account_balance - payment_amount
            is_fully_paid = new_account_balance <= 0

            # Update the account's realtime data
            self.set_account_balance(account['account_id'], new_account_balance)

            # Apply the scheduled payment
            payment_record = {
                "payment_id": str(uuid.uuid4()),
                "account_id": account["account_id"],
                "payment_date": payment_date,
                "previous_balance": prev_account_balance,
                "payment_amount": payment_amount,
                "new_balance": new_account_balance,
                "is_fully_paid": is_fully_paid,
            }
            self.payments.append(payment_record)
            # Record the event
            self.record_event(account, "record_payment", payment_record)

        # Remove the scheduled payments that we just applied from the original list
        other_payments = [p for p in scheduled_payments if p not in scheduled_payments_this_cycle]
        self.accounts_realtime_data[account['account_id']]['scheduled_payments'] = other_payments

    def get_delinquency_checks_for_account(self, account_id):
        # Retrieve delinquency checks for a given account
        return [d for d in self.balance_snapshots if d['account_id'] == account_id]
    
    def get_last_delinquency_check_for_account(self, account_id):
        # Retrieve the last delinquency check for a given account
        checks = self.get_delinquency_checks_for_account(account_id)
        return checks[-1] if checks else None

    def get_scheduled_payments_for_account(self, account_id):
        # Retrieve scheduled payments for a given account
        if account_id in self.accounts_realtime_data:
            return self.accounts_realtime_data[account_id].get('scheduled_payments', [])
        else:
            raise ValueError(f"Account {account_id} not found.")

    def add_scheduled_payment_for_account(self, account_id, scheduled_payment):
        # Add a scheduled payment for a given account
        if account_id in self.accounts_realtime_data:
            if 'scheduled_payments' not in self.accounts_realtime_data[account_id]:
                self.accounts_realtime_data[account_id]['scheduled_payments'] = []
            self.accounts_realtime_data[account_id]['scheduled_payments'].append(scheduled_payment)
        else:
            raise ValueError(f"Account {account_id} not found.")
        
    def get_customer_by_id(self, customer_id):
        # Retrieve the customer associated with a given customer_id
        return next((c for c in self.customers if c['customer_id'] == customer_id), None)

    def get_usages_for_account(self, account_id):
        # Retrieve usages for a given account
        return [u for u in self.usages if u['account_id'] == account_id]
    
    def get_last_usage_for_account(self, account_id):
        # Retrieve the last usage record for a given account
        usages = self.get_usages_for_account(account_id)
        return usages[-1] if usages else None
    
    def get_bills_for_account(self, account_id):
        # Retrieve bills for a given account
        return [b for b in self.billings if b['account_id'] == account_id]
    
    def get_last_bill_for_account(self, account_id):
        # Retrieve the last bill record for a given account
        bills = self.get_bills_for_account(account_id)
        return bills[-1] if bills else None
    
    def get_next_cycle_start_date(self):
        # Calculate the start date for the next cycle
        return self.current_date + pd.Timedelta(days=self.cycle_length)
    
    def set_account_balance(self, account_id, balance):
        # Set the balance for a given account
        if account_id in self.accounts_realtime_data:
            self.accounts_realtime_data[account_id]['balance'] = balance
        else:
            raise ValueError(f"Account {account_id} not found.")
        
    def get_account_balance(self, account_id):
        # Get the balance for a given account
        if account_id in self.accounts_realtime_data:
            return self.accounts_realtime_data[account_id]['balance']
        else:
            raise ValueError(f"Account {account_id} not found.")

    def get_account_status(self, account_id):
        # Get the status for a given account
        if account_id in self.accounts_realtime_data:
            return self.accounts_realtime_data[account_id]['status']
        else:
            raise ValueError(f"Account {account_id} not found.")
        

    def set_account_status(self, account_id, status):
        # Set the status for a given account
        if account_id in self.accounts_realtime_data:
            self.accounts_realtime_data[account_id]['status'] = status
        else:
            raise ValueError(f"Account {account_id} not found.")
        
    
    def get_account_delinquency_status(self, account_id):
        # Get the delinquency status for a given account
        if account_id in self.accounts_realtime_data:
            return self.accounts_realtime_data[account_id]['delinquency_status']
        else:
            raise ValueError(f"Account {account_id} not found.")
        
    def set_account_delinquency_status(self, account_id, status):
        # Set the delinquency status for a given account
        if account_id in self.accounts_realtime_data:
            self.accounts_realtime_data[account_id]['delinquency_status'] = status
        else:
            raise ValueError(f"Account {account_id} not found.")
        
    def get_account_delinquency_score(self, account_id):
        # Get the delinquency score for a given account
        if account_id in self.accounts_realtime_data:
            return self.accounts_realtime_data[account_id]['delinquency_score']
        else:
            raise ValueError(f"Account {account_id} not found.")
        
    def set_account_delinquency_score(self, account_id, score):
        # Set the delinquency score for a given account
        if account_id in self.accounts_realtime_data:
            self.accounts_realtime_data[account_id]['delinquency_score'] = score
        else:
            raise ValueError(f"Account {account_id} not found.")
        
    def get_account_interaction_order(self, account_id):
        # Get the interaction order for a given account
        if account_id in self.accounts_realtime_data:
            return self.accounts_realtime_data[account_id]['interaction_order']
        else:
            raise ValueError(f"Account {account_id} not found.")
        
    def get_payments_last_cycle_for_account(self, account_id):
        # Retrieve payments for the last cycle for a given account
        payments = [p for p in self.payments if p['account_id'] == account_id]
        if not payments:
            return []
        
        cycle_start = self.current_date - pd.Timedelta(days=self.cycle_length)
        cycle_end = self.current_date - pd.Timedelta(days=1)
        
        return [p for p in payments if p['payment_date'] >= cycle_start and p['payment_date'] <= cycle_end]

# Generate a random name for a customer
def random_name():
    fake = faker.Faker()
    return fake.name()


import yaml

def run_simulator_from_config(config_path="config.yaml"):
    """
    Run the UtilitySimulator using parameters from a YAML config file.

    Args:
        config_path (str): Path to the YAML configuration file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    simulator = UtilitySimulator(
        num_customers=config.get("num_customers", 1000),
        num_cycles=config.get("num_cycles", 48),
        cycle_length=config.get("cycle_length", 30),
        start_date=pd.Timestamp(config.get("start_date", "2020-01-01")),
        output_dir=config.get("output_dir", "output"),
        seed=config.get("seed", None),
        description=config.get("description", "Utility Simulation"),
        record_events_flag=config.get("record_events_flag", True),
        settings=config.get("settings", None)
    )
    simulator.run_simulation()
    simulator.export_data(config_dict=config)

# Example usage:
# run_simulator_from_config("config.yaml")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1].endswith(".yaml"):
        run_simulator_from_config(sys.argv[1])
    else:
        simulator = UtilitySimulator(
            num_customers= 1000,
            num_cycles=48,
            cycle_length=30,
            start_date=pd.Timestamp('2020-01-01'),
            description='Utility Simulation with 1000 customers and 48 cycles',
            output_dir='data/utility_data_1k_seed_8',
            seed=8,
            record_events_flag=False,
            settings={
                'customer_reliability_distribution': {
                    'low': 0.5,
                    'medium': 0.3,
                    'high': 0.2
                },
                'usage_ranges': {
                    'low': (100, 350),
                    'medium': (250, 500),
                    'high': (500, 800)
                },
                'plan_rate': {
                    'residential': 0.22,
                    'commercial': 0.35
                },
                'usage_probabilities': {
                    'low': 0.5,
                    'medium': 0.3,
                    'high': 0.2
                },
                'penalty_thresholds': {
                    'suspend': 5,
                    'close': 10
                }
            }
        )
        simulator.run_simulation()
        simulator.export_data()

