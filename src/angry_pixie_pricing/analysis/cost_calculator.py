"""Calculate electricity costs using load profiles and spot prices."""

from typing import Any

import pandas as pd

from angry_pixie_pricing.data.electricity_prices import get_electricity_prices
from angry_pixie_pricing.load_profiles.base import LoadProfile


class CostCalculator:
    """Calculate electricity costs by matching load profiles with spot prices."""

    def __init__(
        self,
        load_profile: LoadProfile,
        region: str,
        vat_rate: float = 0.19,
        grid_fees_kwh: float = 0.0,
        other_fees_kwh: float = 0.0,
        fixed_monthly_fee: float = 0.0,
    ):
        """Initialize cost calculator.

        Args:
            load_profile: Load profile to calculate costs for
            region: Region code for electricity prices
            vat_rate: VAT rate (default 19% for Germany)
            grid_fees_kwh: Grid fees per kWh
            other_fees_kwh: Other fees per kWh (taxes, levies)
            fixed_monthly_fee: Fixed monthly fee
        """
        self.load_profile = load_profile
        self.region = region
        self.vat_rate = vat_rate
        self.grid_fees_kwh = grid_fees_kwh
        self.other_fees_kwh = other_fees_kwh
        self.fixed_monthly_fee = fixed_monthly_fee

        self._price_data: pd.DataFrame | None = None
        self._calculation_results: dict[str, Any] | None = None

    def calculate(self, use_cache: bool = True) -> dict[str, Any]:
        """Calculate total electricity costs.

        Args:
            use_cache: Whether to use cached price data

        Returns:
            Dictionary with cost breakdown and statistics
        """
        # Get electricity prices
        self._price_data = get_electricity_prices(
            self.region,
            self.load_profile.start_date,
            self.load_profile.end_date,
            use_cache=use_cache,
        )

        # Get hourly load profile (prices are hourly)
        hourly_load = self.load_profile.resample_to_hourly()

        # Align data
        aligned_data = self._align_data(hourly_load, self._price_data)

        # Calculate costs
        costs = self._calculate_costs(aligned_data)

        # Add monthly fixed fees
        months = (self.load_profile.end_date - self.load_profile.start_date).days / 30
        costs["fixed_fees"] = self.fixed_monthly_fee * months

        # Calculate totals
        costs["total_before_vat"] = (
            costs["energy_cost"] + costs["grid_fees"] + costs["other_fees"] + costs["fixed_fees"]
        )
        costs["vat"] = costs["total_before_vat"] * self.vat_rate
        costs["total_with_vat"] = costs["total_before_vat"] + costs["vat"]

        # Add statistics
        stats = self._calculate_statistics(aligned_data)

        # Combine results
        self._calculation_results = {
            **costs,
            **stats,
            "period": {
                "start": self.load_profile.start_date,
                "end": self.load_profile.end_date,
                "days": (self.load_profile.end_date - self.load_profile.start_date).days,
            },
        }

        return self._calculation_results

    def _align_data(self, load_data: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        """Align load and price data on common timestamps."""
        # Ensure price data has the expected column
        if "Price" in price_data.columns:
            price_col = "Price"
        elif "price" in price_data.columns:
            price_col = "price"
        else:
            raise ValueError(
                f"Price column not found in data. Available columns: {price_data.columns.tolist()}"
            )

        # Create aligned DataFrame
        aligned = pd.DataFrame(index=load_data.index)
        aligned["power_kw"] = load_data["power_kw"]
        aligned["energy_kwh"] = load_data["energy_kwh"]

        # Convert price from EUR/MWh to EUR/kWh
        aligned["price_eur_kwh"] = price_data[price_col].reindex(aligned.index) / 1000

        # Forward fill any missing prices
        aligned["price_eur_kwh"] = aligned["price_eur_kwh"].ffill()
        aligned["price_eur_kwh"] = aligned["price_eur_kwh"].bfill()

        # Calculate hourly costs
        aligned["energy_cost_eur"] = aligned["energy_kwh"] * aligned["price_eur_kwh"]

        return aligned

    def _calculate_costs(self, data: pd.DataFrame) -> dict[str, float]:
        """Calculate cost components."""
        total_energy = data["energy_kwh"].sum()

        return {
            "energy_cost": data["energy_cost_eur"].sum(),
            "grid_fees": total_energy * self.grid_fees_kwh,
            "other_fees": total_energy * self.other_fees_kwh,
            "total_consumption_kwh": total_energy,
        }

    def _calculate_statistics(self, data: pd.DataFrame) -> dict[str, Any]:
        """Calculate additional statistics."""
        # Price statistics
        price_stats = {
            "avg_price_eur_kwh": data["price_eur_kwh"].mean(),
            "min_price_eur_kwh": data["price_eur_kwh"].min(),
            "max_price_eur_kwh": data["price_eur_kwh"].max(),
            "weighted_avg_price_eur_kwh": (
                (data["price_eur_kwh"] * data["energy_kwh"]).sum() / data["energy_kwh"].sum()
            ),
        }

        # Load statistics
        load_stats = self.load_profile.get_statistics()

        # Time-based analysis
        data["hour"] = data.index.hour
        data["weekday"] = data.index.weekday < 5

        # Average costs by time of day
        hourly_avg = data.groupby("hour").agg({"energy_kwh": "sum", "energy_cost_eur": "sum"})
        hourly_avg["avg_cost_per_kwh"] = hourly_avg["energy_cost_eur"] / hourly_avg["energy_kwh"]

        # Weekday vs weekend
        weekday_stats = data.groupby("weekday").agg({"energy_kwh": "sum", "energy_cost_eur": "sum"})

        return {
            **price_stats,
            "load_stats": load_stats,
            "hourly_distribution": hourly_avg.to_dict(),
            "weekday_consumption_kwh": weekday_stats.loc[True, "energy_kwh"]
            if True in weekday_stats.index
            else 0,
            "weekend_consumption_kwh": weekday_stats.loc[False, "energy_kwh"]
            if False in weekday_stats.index
            else 0,
            "peak_cost_hour": hourly_avg["avg_cost_per_kwh"].idxmax(),
            "lowest_cost_hour": hourly_avg["avg_cost_per_kwh"].idxmin(),
        }

    def get_detailed_breakdown(self) -> pd.DataFrame:
        """Get hourly breakdown of consumption and costs."""
        if self._calculation_results is None:
            self.calculate()

        # Prepare detailed hourly data
        hourly_load = self.load_profile.resample_to_hourly()
        aligned_data = self._align_data(hourly_load, self._price_data)

        # Add additional calculated columns
        aligned_data["grid_fees_eur"] = aligned_data["energy_kwh"] * self.grid_fees_kwh
        aligned_data["other_fees_eur"] = aligned_data["energy_kwh"] * self.other_fees_kwh
        aligned_data["total_cost_eur"] = (
            aligned_data["energy_cost_eur"]
            + aligned_data["grid_fees_eur"]
            + aligned_data["other_fees_eur"]
        )

        return aligned_data

    def compare_with_flat_rate(self, flat_rate_eur_kwh: float) -> dict[str, Any]:
        """Compare spot market costs with a flat rate tariff.

        Args:
            flat_rate_eur_kwh: Flat rate price per kWh

        Returns:
            Comparison results
        """
        if self._calculation_results is None:
            self.calculate()

        total_kwh = self._calculation_results["total_consumption_kwh"]

        # Calculate flat rate costs
        flat_energy_cost = total_kwh * flat_rate_eur_kwh
        flat_total_before_vat = (
            flat_energy_cost
            + self._calculation_results["grid_fees"]
            + self._calculation_results["other_fees"]
            + self._calculation_results["fixed_fees"]
        )
        flat_vat = flat_total_before_vat * self.vat_rate
        flat_total_with_vat = flat_total_before_vat + flat_vat

        # Calculate savings
        spot_total = self._calculation_results["total_with_vat"]
        savings = flat_total_with_vat - spot_total
        savings_percent = (savings / flat_total_with_vat) * 100

        return {
            "flat_rate_eur_kwh": flat_rate_eur_kwh,
            "flat_rate_total": flat_total_with_vat,
            "spot_market_total": spot_total,
            "savings_eur": savings,
            "savings_percent": savings_percent,
            "better_option": "spot_market" if savings > 0 else "flat_rate",
        }

    def optimize_load_shifting(
        self, shiftable_percentage: float = 0.2, _max_shift_hours: int = 4
    ) -> dict[str, Any]:
        """Calculate potential savings from load shifting.

        Args:
            shiftable_percentage: Percentage of load that can be shifted
            max_shift_hours: Maximum hours load can be shifted

        Returns:
            Optimization results
        """
        if self._calculation_results is None:
            self.calculate()

        # Get detailed hourly data
        hourly_data = self.get_detailed_breakdown()

        # Identify shiftable load
        total_load = hourly_data["energy_kwh"].sum()
        shiftable_load = total_load * shiftable_percentage

        # Find cheapest hours within shifting windows
        original_cost = hourly_data["energy_cost_eur"].sum()

        # Simple optimization: move load to cheapest hours
        sorted_hours = hourly_data.sort_values("price_eur_kwh")
        cheapest_hours = sorted_hours.head(int(len(sorted_hours) * shiftable_percentage))

        # Calculate potential savings
        avg_original_price = hourly_data["energy_cost_eur"].sum() / hourly_data["energy_kwh"].sum()
        avg_optimized_price = cheapest_hours["price_eur_kwh"].mean()

        potential_savings = shiftable_load * (avg_original_price - avg_optimized_price)

        return {
            "shiftable_load_kwh": shiftable_load,
            "original_cost": original_cost,
            "potential_savings": potential_savings,
            "savings_percent": (potential_savings / original_cost) * 100,
            "cheapest_hours": cheapest_hours.index.hour.unique().tolist(),
            "avg_price_reduction": avg_original_price - avg_optimized_price,
        }
