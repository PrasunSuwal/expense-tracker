import {
  LuLayoutDashboard,
  LuHandCoins,
  LuWalletMinimal,
  LuLogOut,
  LuActivity,
  LuUser,
} from "react-icons/lu";

export const SIDE_MENU_DATA = [
  {
    id: "01",
    label: "Dashboard",
    icon: LuLayoutDashboard,
    path: "/dashboard",
  },
  {
    id: "02",
    label: "Income",
    icon: LuWalletMinimal,
    path: "/income",
  },
  {
    id: "03",
    label: "Expense",
    icon: LuHandCoins,
    path: "/expense",
  },
  {
    id: "04",
    label: "Analysis",
    icon: LuActivity,
    path: "/analysis",
  },
  {
    id: "05",
    label: "Profile",
    icon: LuUser,
    path: "/profile",
  },
  {
    id: "06",
    label: "LogOut",
    icon: LuLogOut,
    path: "logout",
  },
];
