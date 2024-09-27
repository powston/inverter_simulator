class Battery:
    def __init__(self, capacity: float, charge_rate: float, initial_charge: float = None, loss_rate: float = 5):
        self.capacity = capacity
        self.charge_rate = charge_rate
        self.charge = initial_charge if initial_charge is not None else capacity / 2
        self.loss_rate = loss_rate

    @property
    def soc(self) -> float:
        return (self.charge / self.capacity) * 100

    def charge_battery(self, amount: float, interval: int = 5) -> float:
        per_hour = 60 / interval
        charge_ability = min(self.charge_rate, max(0, self.capacity - self.charge) * per_hour)
        actual_charge = max(0, min(amount, charge_ability))
        charge_minus_loss = actual_charge * ((100 - self.loss_rate) / 100)
        self.charge = min(self.capacity, self.charge + (charge_minus_loss / 12))
        return actual_charge

    def discharge_battery(self, amount: float, interval: int = 5) -> float:
        per_hour = 60 / interval
        discharge_ability = min(self.charge_rate, max(0, self.charge) * per_hour)
        actual_discharge = max(0, min(amount, discharge_ability))
        discharge_plus_loss = actual_discharge * ((100 + self.loss_rate) / 100)
        self.charge = max(0, self.charge - (discharge_plus_loss / 12))
        return actual_discharge

    def reset(self) -> None:
        self.charge = self.capacity / 2