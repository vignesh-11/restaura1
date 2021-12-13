import Index from "views/Index.js";
import Maps from "views/examples/Maps.js";
import Register from "views/examples/Register.js";
import Login from "views/examples/Login.js";
import Landing from "views/examples/Landing"
import MultiStepForm from "views/examples/MultiStepForm";
import Reports from "views/examples/Reports";
import Tables from "views/examples/Tables";

var routes = [
  {
    path: "/restaurant",
    name: "Restaurant",
    icon: "ni ni-tv-2 text-primary",
    component: Index,
    layout: "/customer",
  },
  {
    path: "/leaderboard",
    name: "Leaderboard",
    icon: "ni ni-sound-wave",
    component: Tables,
    layout: "/customer",
  },
  {
    path: "/maps",
    name: "Map",
    icon: "ni ni-pin-3 text-orange",
    component: Maps,
    layout: "/customer",
  },
  {
    path: "/login",
    name: "Login",
    icon: "ni ni-key-25 text-info",
    component: Login,
    layout: "/auth",
  },
  {
    path: "/register",
    name: "Register",
    icon: "ni ni-circle-08 text-pink",
    component: Register,
    layout: "/auth",
  },
  {
    path: "/landing",
    name: "Landing",
    icon: "ni ni-circle-08 text-pink",
    component: Landing,
    layout: "/auth",
  },
  {
    path: "/reports",
    name: "Reports",
    icon: "ni ni-collection",
    component: Reports,
    layout: "/business",
  },
  {
    path: "/open",
    name: "Open Restaurant",
    icon: "ni ni-basket text-orange",
    component: MultiStepForm,
    layout: "/business",
  },
];
export default routes;
