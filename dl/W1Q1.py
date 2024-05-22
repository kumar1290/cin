from datetime import datetime

class UserPreferences:
    def __init__(self, preferred_temperature=70, is_home=True):
        self.preferred_temperature = preferred_temperature
        self.is_home = is_home

class SmartThermostat:
    def __init__(self, user_preferences):
        self.user_preferences = user_preferences
        self.current_temperature = 70

    def get_external_temperature(self):
        return 70

    def get_current_time(self):
        return datetime.now()

    def adjust_thermostat(self):
        current_time = self.get_current_time()
        external_temperature = self.get_external_temperature()
        user_pref_temp = self.user_preferences.preferred_temperature

        if current_time.hour >= 6 and current_time.hour < 8 and external_temperature < 65:
            self.current_temperature = 70
        elif current_time.hour >= 8 and current_time.hour < 17 and external_temperature > 75:
            self.current_temperature = 72
        elif (current_time.hour >= 17 or current_time.hour < 6) and external_temperature < 60:
            self.current_temperature = 65
        elif self.user_preferences.is_home:
            self.current_temperature = user_pref_temp
        else:
            if external_temperature < 65:
                self.current_temperature = 60
            else:
                self.current_temperature = 80

        print(f"Thermostat adjusted to {self.current_temperature}°F")

user_preferences = UserPreferences(preferred_temperature=75, is_home=True)
thermostat = SmartThermostat(user_preferences)
thermostat.adjust_thermostat()
