import {
  LuLayoutDashboard,
  LuHandCoins,
  LuWalletMinimal,
  LuLogOut,
  LuHome,
} from "react-icons/lu";

export const SIDE_MENU_DATA = [
  {
    id: "00",
    label: "Home",
    icon: LuHome,
    path: "/",
  },
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
    id: "06",
    label: "LogOut",
    icon: LuLogOut,
    path: "logout",
  },
];
