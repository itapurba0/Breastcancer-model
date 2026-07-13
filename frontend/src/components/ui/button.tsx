import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-3xl text-sm font-semibold ring-offset-background transition-transform duration-300 ease-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default: "bg-highlight text-primary-foreground hover:bg-highlight/95 soft-shadow-sm hover:soft-shadow-md hover:-translate-y-1",
        destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/90 soft-shadow-sm hover:soft-shadow-md",
        outline: "border-2 border-brand bg-transparent text-brand hover:bg-brand/5 hover:text-brand hover:-translate-y-1",
        secondary: "bg-brand text-primary-foreground hover:bg-brand/95 soft-shadow-sm hover:soft-shadow-md",
        ghost: "hover:bg-sage hover:text-accent-foreground rounded-3xl",
        link: "text-brand underline-offset-4 hover:underline",
        medical: "bg-gradient-to-r from-brand to-brand/85 text-primary-foreground soft-shadow-sm hover:soft-shadow-md hover:-translate-y-1",
        success: "bg-sage text-accent-foreground hover:bg-sage/95 soft-shadow-sm hover:-translate-y-1",
      },
      size: {
        default: "h-12 px-6",
        sm: "h-10 rounded-2xl px-4",
        lg: "h-14 rounded-3xl px-8 text-base",
        icon: "h-12 w-12 rounded-3xl",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
  VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return <Comp className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />;
  },
);
Button.displayName = "Button";

export { Button, buttonVariants };
